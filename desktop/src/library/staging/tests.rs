use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::watch;

use super::*;

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    instance: Uuid,
    control: CopyControl,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let instance = new_import_id().unwrap();
        let fixture = Self {
            _directory: directory,
            root,
            instance,
            control: CopyControl {
                cancelled: AtomicBool::new(false),
                closing: watch::channel(false).1,
            },
        };
        fixture.add_instance(instance);
        fixture
    }

    fn stage_dir(&self, instance: Uuid) -> PathBuf {
        self.root
            .join("knowledge/temporary")
            .join(instance.to_string())
    }

    fn add_instance(&self, instance: Uuid) {
        let directory = self.stage_dir(instance);
        fs::create_dir_all(&directory).unwrap();
        for path in directory.ancestors().take(3) {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }
    }

    fn prepare(&self, path: &Path) -> Result<(HostImport, Arc<StagedFile>), &'static str> {
        StagedFile::prepare(&self.root, self.instance, path, None, &self.control)
    }

    fn assert_empty(&self, instance: Uuid) {
        assert!(fs::read_dir(self.stage_dir(instance))
            .unwrap()
            .next()
            .is_none());
    }
}

#[test]
fn published_snapshot_is_private_and_restaged_with_the_same_uuid_after_unlink() {
    let fixture = Fixture::new();
    let source = fixture.root.join("教材.MD");
    // Split a four-byte character across the Markdown validation buffer.
    let contents = format!("{}📚\n", "a".repeat(BLOCK_BYTES));
    fs::write(&source, &contents).unwrap();
    let (request, staged) = fixture.prepare(&source).unwrap();
    assert_eq!(request.import_id.get_version(), Some(uuid::Version::Random));
    let name = request.import_id.to_string();
    let first_stage = fixture.stage_dir(fixture.instance).join(&name);
    assert_eq!(first_stage.metadata().unwrap().mode() & 0o777, 0o600);
    assert_eq!(fs::read_to_string(&first_stage).unwrap(), contents);
    assert!(Arc::ptr_eq(
        &staged,
        &staged
            .for_instance(
                &fixture.root,
                fixture.instance,
                request.import_id,
                &fixture.control
            )
            .unwrap(),
    ));

    // Python consumed the entry, and the originally selected file has been edited.
    fs::remove_file(first_stage).unwrap();
    fs::write(source, "different contents").unwrap();
    let next_instance = new_import_id().unwrap();
    fixture.add_instance(next_instance);
    let replacement = staged
        .for_instance(
            &fixture.root,
            next_instance,
            request.import_id,
            &fixture.control,
        )
        .unwrap();
    assert_eq!(
        fs::read_to_string(fixture.stage_dir(next_instance).join(name)).unwrap(),
        contents
    );
    drop(replacement);
    fixture.assert_empty(next_instance);
}

#[test]
fn invalid_sources_fail_before_publishing_anything() {
    let fixture = Fixture::new();
    let valid = fixture.root.join("not-markdown.pdf");
    fs::write(&valid, "# fake PDF").unwrap();
    assert_eq!(
        fixture.prepare(&valid).err(),
        Some("INVALID_TEXTBOOK_FORMAT")
    );
    let link = fixture.root.join("link.md");
    symlink(&valid, &link).unwrap();
    assert_eq!(fixture.prepare(&link).err(), Some(INVALID_SOURCE));
    let parent_link = fixture.root.join("linked");
    symlink(&fixture.root, &parent_link).unwrap();
    assert_eq!(
        fixture.prepare(&parent_link.join("not-markdown.pdf")).err(),
        Some(INVALID_SOURCE)
    );

    let fifo = fixture.root.join("fifo.md");
    let c_path = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    assert_eq!(fixture.prepare(&fifo).err(), Some(INVALID_SOURCE));

    let oversized = fixture.root.join("big.md");
    File::create(&oversized)
        .unwrap()
        .set_len(MAX_BYTES + 1)
        .unwrap();
    assert_eq!(fixture.prepare(&oversized).err(), Some(TOO_LARGE));
    let binary = fixture.root.join("binary.md");
    fs::write(&binary, [b'#', 0, 0xff]).unwrap();
    assert_eq!(
        fixture.prepare(&binary).err(),
        Some("INVALID_TEXTBOOK_FORMAT")
    );
    fixture.assert_empty(fixture.instance);
}

#[test]
fn source_revision_and_staging_directory_are_checked() {
    let fixture = Fixture::new();
    let path = fixture.root.join("source.md");
    fs::write(&path, "# source").unwrap();
    let source = SelectedFile::open(&path).unwrap();
    fs::write(&path, "# other contents").unwrap();
    assert_eq!(source.verify(), Err(SOURCE_CHANGED));
    fs::rename(&path, fixture.root.join("moved.md")).unwrap();
    fs::write(&path, "# replacement").unwrap();
    assert_eq!(source.verify(), Err(SOURCE_CHANGED));

    let directory = fixture.stage_dir(fixture.instance);
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(fixture.prepare(&path).err(), Some(STAGING_UNAVAILABLE));
    fs::remove_dir(&directory).unwrap();
    symlink(&fixture.root, &directory).unwrap();
    assert_eq!(fixture.prepare(&path).err(), Some(STAGING_UNAVAILABLE));
}

#[test]
fn copy_failure_and_cancellation_leave_no_staging_entry() {
    struct DiskFull;
    impl Write for DiskFull {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from_raw_os_error(libc::ENOSPC))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct CancelAfterWrite<'a>(&'a CopyControl);
    impl Write for CancelAfterWrite<'_> {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.0.cancelled.store(true, Ordering::Relaxed);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let fixture = Fixture::new();
    let entry =
        StagingEntry::create(open_staging(&fixture.root, fixture.instance).unwrap()).unwrap();
    assert_eq!(
        copy_blocks(&mut &b"123"[..], &mut DiskFull, 3, &fixture.control),
        Err("IMPORT_DISK_FULL")
    );
    assert_eq!(
        copy_blocks(&mut &b"123"[..], &mut io::sink(), 4, &fixture.control),
        Err(SOURCE_CHANGED)
    );
    let contents = vec![b'x'; 2 * BLOCK_BYTES];
    let mut writer = CancelAfterWrite(&fixture.control);
    assert_eq!(
        copy_blocks(
            &mut contents.as_slice(),
            &mut writer,
            contents.len() as u64,
            &fixture.control
        ),
        Err("IMPORT_CANCELLED")
    );
    drop(entry);
    fixture.assert_empty(fixture.instance);
}
