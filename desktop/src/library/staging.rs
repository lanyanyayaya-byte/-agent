use std::ffi::{CStr, CString};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{native::CopyControl, HostImport};

mod format;
use format::TextbookFormat;

const MAX_BYTES: u64 = 100 * 1024 * 1024;
const BLOCK_BYTES: usize = 256 * 1024;
const INVALID_SOURCE: &str = "INVALID_IMPORT_SOURCE";
const SOURCE_CHANGED: &str = "IMPORT_SOURCE_CHANGED";
const STAGING_UNAVAILABLE: &str = "IMPORT_STAGING_UNAVAILABLE";
const COPY_FAILED: &str = "IMPORT_COPY_FAILED";
const TOO_LARGE: &str = "TEXTBOOK_TOO_LARGE";

#[derive(PartialEq, Eq)]
struct Revision {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl From<Metadata> for Revision {
    fn from(info: Metadata) -> Self {
        Self {
            device: info.dev(),
            inode: info.ino(),
            mode: info.mode(),
            links: info.nlink(),
            size: info.len(),
            modified: (info.mtime(), info.mtime_nsec()),
            changed: (info.ctime(), info.ctime_nsec()),
        }
    }
}

struct SelectedFile<'a> {
    path: &'a Path,
    file: File,
    revision: Revision,
}

impl<'a> SelectedFile<'a> {
    fn open(path: &'a Path) -> Result<Self, &'static str> {
        let before = fs::symlink_metadata(path).map_err(|_| INVALID_SOURCE)?;
        if !path.is_absolute() || !before.is_file() {
            return Err(INVALID_SOURCE);
        }
        if before.len() > MAX_BYTES {
            return Err(TOO_LARGE);
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW_ANY | libc::O_NONBLOCK)
            .open(path)
            .map_err(|_| INVALID_SOURCE)?;
        let selected = Self {
            path,
            file,
            revision: before.into(),
        };
        selected.verify()?;
        Ok(selected)
    }

    fn verify(&self) -> Result<(), &'static str> {
        let current = fs::symlink_metadata(self.path).map_err(|_| SOURCE_CHANGED)?;
        let opened = self.file.metadata().map_err(|_| SOURCE_CHANGED)?;
        let mut descriptor_path = [0_u8; libc::MAXPATHLEN as usize];
        // F_GETPATH writes a NUL-terminated filesystem path into a MAXPATHLEN buffer.
        if unsafe {
            libc::fcntl(
                self.file.as_raw_fd(),
                libc::F_GETPATH,
                descriptor_path.as_mut_ptr(),
            )
        } < 0
            || CStr::from_bytes_until_nul(&descriptor_path).map(CStr::to_bytes)
                != Ok(self.path.as_os_str().as_bytes())
            || Revision::from(current) != self.revision
            || Revision::from(opened) != self.revision
        {
            return Err(SOURCE_CHANGED);
        }
        Ok(())
    }
}

// The writable entry starts under a private scratch name. Drop removes only the
// entry still pointing to our descriptor, including after a failed publish.
struct StagingEntry {
    directory: File,
    name: CString,
    file: File,
}

impl StagingEntry {
    fn create(directory: File) -> Result<Self, &'static str> {
        let name = CString::new(format!(".copy-{}", new_import_id()?)).unwrap();
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600 as libc::c_int,
            )
        };
        if fd < 0 {
            return Err(copy_error(io::Error::last_os_error()));
        }
        let entry = Self {
            directory,
            name,
            file: unsafe { File::from_raw_fd(fd) },
        };
        entry
            .file
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(copy_error)?;
        Ok(entry)
    }

    fn publish(mut self, import_id: Uuid, current_directory: &File) -> Result<Self, &'static str> {
        let original = self.directory.metadata().map_err(copy_error)?;
        let current = current_directory.metadata().map_err(copy_error)?;
        if (original.dev(), original.ino()) != (current.dev(), current.ino())
            || !self.is_linked()?
        {
            return Err(STAGING_UNAVAILABLE);
        }
        let name = CString::new(import_id.to_string()).unwrap();
        let directory = self.directory.as_raw_fd();
        if unsafe {
            libc::renameatx_np(
                directory,
                self.name.as_ptr(),
                directory,
                name.as_ptr(),
                libc::RENAME_EXCL,
            )
        } != 0
        {
            return Err(copy_error(io::Error::last_os_error()));
        }
        self.name = name;
        self.directory.sync_all().map_err(copy_error)?;
        Ok(self)
    }

    fn is_linked(&self) -> Result<bool, &'static str> {
        let opened = self.file.metadata().map_err(copy_error)?;
        let mut linked = std::mem::MaybeUninit::<libc::stat>::uninit();
        let result = unsafe {
            libc::fstatat(
                self.directory.as_raw_fd(),
                self.name.as_ptr(),
                linked.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if result == 0 {
            let linked = unsafe { linked.assume_init() };
            if (linked.st_dev as u64, linked.st_ino as u64) != (opened.dev(), opened.ino()) {
                return Err(STAGING_UNAVAILABLE);
            }
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::NotFound {
            return Ok(false);
        }
        Err(copy_error(error))
    }
}

impl Drop for StagingEntry {
    fn drop(&mut self) {
        if self.is_linked() == Ok(true) {
            unsafe { libc::unlinkat(self.directory.as_raw_fd(), self.name.as_ptr(), 0) };
        }
    }
}

pub(super) struct StagedFile {
    instance: Uuid,
    entry: StagingEntry,
    revision: Revision,
    sha256: [u8; 32],
}

impl StagedFile {
    pub(super) fn prepare(
        app_data: &Path,
        instance: Uuid,
        source_path: &Path,
        document_id: Option<String>,
        control: &CopyControl,
    ) -> Result<(HostImport, Arc<Self>), &'static str> {
        control.check()?;
        let filename = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("VALIDATION_ERROR")?;
        let format = TextbookFormat::from_filename(filename)?;
        let mut source = SelectedFile::open(source_path)?;
        let request = HostImport {
            import_id: new_import_id()?,
            filename: filename.to_owned(),
            document_id,
        };
        let mut entry = StagingEntry::create(open_staging(app_data, instance)?)?;
        let sha256 = copy_blocks(
            &mut source.file,
            &mut entry.file,
            source.revision.size,
            control,
        )?;
        format.validate(&mut entry.file, source.revision.size, control)?;
        entry.file.sync_all().map_err(copy_error)?;
        source.verify()?;
        control.check()?;
        let staged = Self::publish(entry, app_data, instance, request.import_id, sha256)?;
        Ok((request, staged))
    }

    pub(super) fn for_instance(
        self: &Arc<Self>,
        app_data: &Path,
        instance: Uuid,
        import_id: Uuid,
        control: &CopyControl,
    ) -> Result<Arc<Self>, &'static str> {
        control.check()?;
        let directory = open_staging(app_data, instance)?;
        if self.instance == instance && self.entry.is_linked()? {
            let current = directory.metadata().map_err(copy_error)?;
            let previous = self.entry.directory.metadata().map_err(copy_error)?;
            let file = self.entry.file.metadata().map_err(copy_error)?;
            if (current.dev(), current.ino()) != (previous.dev(), previous.ino()) {
                return Err(STAGING_UNAVAILABLE);
            }
            if Revision::from(file) == self.revision {
                return Ok(self.clone());
            }
            // A concurrent request can consume the entry while we check it.
            if self.entry.is_linked()? {
                return Err(SOURCE_CHANGED);
            }
        }

        // Even if Python has consumed the staging name, the descriptor still
        // holds the exact bytes for this import. Verify them before republishing.
        let mut entry = StagingEntry::create(directory)?;
        let mut source = &self.entry.file;
        source.rewind().map_err(copy_error)?;
        let sha256 = copy_blocks(&mut source, &mut entry.file, self.revision.size, control)?;
        if sha256 != self.sha256 {
            return Err(SOURCE_CHANGED);
        }
        entry.file.sync_all().map_err(copy_error)?;
        control.check()?;
        Self::publish(entry, app_data, instance, import_id, sha256)
    }

    fn publish(
        entry: StagingEntry,
        app_data: &Path,
        instance: Uuid,
        import_id: Uuid,
        sha256: [u8; 32],
    ) -> Result<Arc<Self>, &'static str> {
        let entry = entry.publish(import_id, &open_staging(app_data, instance)?)?;
        let info = entry.file.metadata().map_err(copy_error)?;
        if !info.is_file()
            || info.nlink() != 1
            || info.mode() & 0o7777 != 0o600
            || info.uid() != unsafe { libc::geteuid() }
            || !entry.is_linked()?
        {
            return Err(STAGING_UNAVAILABLE);
        }
        let revision = info.into();
        Ok(Arc::new(Self {
            instance,
            entry,
            revision,
            sha256,
        }))
    }
}

fn open_staging(app_data: &Path, instance: Uuid) -> Result<File, &'static str> {
    let mut directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW_ANY)
        .open(app_data)
        .map_err(|_| STAGING_UNAVAILABLE)?;
    for component in ["knowledge", "temporary", &instance.to_string()] {
        let name = CString::new(component).unwrap();
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(STAGING_UNAVAILABLE);
        }
        directory = unsafe { File::from_raw_fd(fd) };
        let info = directory.metadata().map_err(|_| STAGING_UNAVAILABLE)?;
        if info.mode() & 0o7777 != 0o700 || info.uid() != unsafe { libc::geteuid() } {
            return Err(STAGING_UNAVAILABLE);
        }
    }
    Ok(directory)
}

fn new_import_id() -> Result<Uuid, &'static str> {
    let mut bytes = [0; 16];
    getrandom::fill(&mut bytes).map_err(|_| "IMPORT_ID_GENERATION_FAILED")?;
    Ok(uuid::Builder::from_random_bytes(bytes).into_uuid())
}

fn copy_blocks(
    source: &mut impl Read,
    destination: &mut impl Write,
    expected: u64,
    control: &CopyControl,
) -> Result<[u8; 32], &'static str> {
    let mut buffer = vec![0; BLOCK_BYTES];
    let mut digest = Sha256::new();
    let mut total = 0;
    loop {
        control.check()?;
        let bytes = match source.read(&mut buffer) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result.map_err(copy_error)?,
        };
        if bytes == 0 {
            break;
        }
        total += bytes as u64;
        if total > MAX_BYTES {
            return Err(TOO_LARGE);
        }
        if total > expected {
            return Err(SOURCE_CHANGED);
        }
        destination
            .write_all(&buffer[..bytes])
            .map_err(copy_error)?;
        digest.update(&buffer[..bytes]);
    }
    if total != expected {
        return Err(SOURCE_CHANGED);
    }
    Ok(digest.finalize().into())
}

fn copy_error(error: io::Error) -> &'static str {
    match error.raw_os_error() {
        Some(libc::ENOSPC | libc::EDQUOT) => "IMPORT_DISK_FULL",
        _ => COPY_FAILED,
    }
}

#[cfg(test)]
mod tests;
