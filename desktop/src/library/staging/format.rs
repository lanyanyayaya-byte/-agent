use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use super::{copy_error, CopyControl, BLOCK_BYTES};

const INVALID: &str = "INVALID_TEXTBOOK_FORMAT";

pub(super) enum TextbookFormat {
    Pdf,
    Docx,
    Markdown,
}

impl TextbookFormat {
    pub(super) fn from_filename(name: &str) -> Result<Self, &'static str> {
        if name.chars().count() > 255
            || name
                .chars()
                .any(|c| matches!(c, '/' | '\\' | '\x00'..='\x1f' | '\x7f'))
        {
            return Err("VALIDATION_ERROR");
        }
        let (stem, extension) = name.rsplit_once('.').ok_or(INVALID)?;
        if stem.trim().is_empty() {
            return Err(INVALID);
        }
        match extension.to_ascii_lowercase().as_str() {
            "pdf" => Ok(Self::Pdf),
            "docx" => Ok(Self::Docx),
            "md" => Ok(Self::Markdown),
            _ => Err(INVALID),
        }
    }

    pub(super) fn validate(
        &self,
        file: &mut File,
        size: u64,
        control: &CopyControl,
    ) -> Result<(), &'static str> {
        if size == 0 {
            return Err(INVALID);
        }
        file.rewind().map_err(copy_error)?;
        match self {
            Self::Pdf => pdf(file, size),
            Self::Docx => docx(file, size, control),
            Self::Markdown => markdown(file, control),
        }
    }
}

fn read_exact(file: &mut File, slice: &mut [u8]) -> Result<(), &'static str> {
    file.read_exact(slice).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            INVALID
        } else {
            copy_error(error)
        }
    })
}

fn tail(file: &mut File, size: u64, limit: u64) -> Result<Vec<u8>, &'static str> {
    let length = size.min(limit);
    file.seek(SeekFrom::Start(size - length))
        .map_err(copy_error)?;
    let mut contents = vec![0; length as usize];
    read_exact(file, &mut contents)?;
    Ok(contents)
}

fn pdf(file: &mut File, size: u64) -> Result<(), &'static str> {
    let mut header = [0; 8];
    read_exact(file, &mut header)?;
    let tail = tail(file, size, 1024)?;
    let end = tail.iter().rposition(|byte| !b" \t\r\n\x0c".contains(byte));
    let trimmed = &tail[..end.map_or(0, |position| position + 1)];
    if (header.starts_with(b"%PDF-1.") && matches!(header[7], b'0'..=b'7')
        || &header == b"%PDF-2.0")
        && trimmed.ends_with(b"%%EOF")
    {
        return Ok(());
    }
    Err(INVALID)
}

fn markdown(file: &mut File, control: &CopyControl) -> Result<(), &'static str> {
    let mut buffer = vec![0; BLOCK_BYTES + 3];
    let mut carry = 0;
    loop {
        control.check()?;
        let bytes = match file.read(&mut buffer[carry..]) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result.map_err(copy_error)?,
        };
        if bytes == 0 {
            return if carry == 0 { Ok(()) } else { Err(INVALID) };
        }
        let end = carry + bytes;
        let text = &buffer[..end];
        if text.contains(&0) {
            return Err(INVALID);
        }
        carry = match std::str::from_utf8(text) {
            Ok(_) => 0,
            Err(error) if error.error_len().is_some() => return Err(INVALID),
            Err(error) => {
                let start = error.valid_up_to();
                buffer.copy_within(start..end, 0);
                end - start
            }
        };
    }
}

// Inspect only the bounded ZIP central directory. A generic ZIP is not a
// DOCX. The Python service independently verifies the staged file.
fn docx(file: &mut File, size: u64, control: &CopyControl) -> Result<(), &'static str> {
    const END_BYTES: usize = 22;
    const MAX_DIRECTORY: usize = 4 * 1024 * 1024;
    const MAX_ENTRIES: usize = 10_000;
    const MAX_EXPANDED_BYTES: u64 = 512 * 1024 * 1024;

    let mut signature = [0; 4];
    read_exact(file, &mut signature)?;
    if &signature != b"PK\x03\x04" {
        return Err(INVALID);
    }
    let tail = tail(file, size, 65_535 + END_BYTES as u64)?;
    let end = tail.len().checked_sub(END_BYTES).ok_or(INVALID)?;
    let index = if tail[end..].starts_with(b"PK\x05\x06") && tail.ends_with(b"\0\0") {
        end
    } else {
        tail.windows(4)
            .rposition(|bytes| bytes == b"PK\x05\x06")
            .ok_or(INVALID)?
    };
    if index > end {
        return Err(INVALID);
    }
    let record = &tail[index..];
    let entries = short(record, 10);
    let length = word(record, 12) as usize;
    let offset = u64::from(word(record, 16));
    if short(record, 4) != 0
        || short(record, 6) != 0
        || short(record, 8) != entries
        || !(1..=MAX_ENTRIES).contains(&entries)
        || length > MAX_DIRECTORY
        || offset + length as u64 != size - tail.len() as u64 + index as u64
        || END_BYTES + short(record, 20) != record.len()
    {
        return Err(INVALID);
    }

    file.seek(SeekFrom::Start(offset)).map_err(copy_error)?;
    let mut directory = vec![0; length];
    read_exact(file, &mut directory)?;
    let mut entries_data = directory.as_slice();
    let mut required = [false; 3];
    let mut expanded = 0;
    for _ in 0..entries {
        control.check()?;
        if entries_data.len() < 46 || !entries_data.starts_with(b"PK\x01\x02") {
            return Err(INVALID);
        }
        let name_length = short(entries_data, 28);
        let next = 46 + name_length + short(entries_data, 30) + short(entries_data, 32);
        let unpacked = word(entries_data, 24);
        expanded += u64::from(unpacked);
        if next > entries_data.len()
            || short(entries_data, 8) & 1 != 0
            || expanded > MAX_EXPANDED_BYTES
        {
            return Err(INVALID);
        }
        match &entries_data[46..46 + name_length] {
            b"[Content_Types].xml" => required[0] = unpacked > 0,
            b"_rels/.rels" => required[1] = unpacked > 0,
            b"word/document.xml" => required[2] = unpacked > 0,
            _ => {}
        }
        entries_data = &entries_data[next..];
    }
    if !entries_data.is_empty() || !required.into_iter().all(|found| found) {
        return Err(INVALID);
    }
    Ok(())
}

fn short(bytes: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
