use tauri::WebviewWindow;
use tokio::sync::Mutex;

#[derive(Default)]
pub(crate) struct NativePicker(pub(crate) Mutex<()>);

pub(crate) enum Selection {
    GrantFile,
    GrantDirectory,
    #[cfg(target_os = "macos")]
    Textbook,
}

#[cfg(target_os = "macos")]
pub(crate) async fn pick(
    window: WebviewWindow,
    selection: Selection,
    failed: &'static str,
) -> Result<Option<String>, &'static str> {
    use std::cell::Cell;

    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSModalResponseCancel, NSModalResponseOK, NSOpenPanel, NSWindow};
    use objc2_foundation::{NSArray, NSString};
    use objc2_uniform_type_identifiers::UTType;
    use tokio::sync::oneshot;

    let (reply, result) = oneshot::channel();
    let parent = window.clone();
    window
        .run_on_main_thread(move || {
            let (Some(main_thread), Ok(parent)) = (MainThreadMarker::new(), parent.ns_window())
            else {
                let _ = reply.send(Err(failed));
                return;
            };
            // Tauri supplies its NSWindow and AppKit calls the completion on the main thread.
            let Some(parent) = (unsafe { parent.cast::<NSWindow>().as_ref() }) else {
                let _ = reply.send(Err(failed));
                return;
            };
            let panel = NSOpenPanel::openPanel(main_thread);
            let file = !matches!(selection, Selection::GrantDirectory);
            panel.setCanChooseFiles(file);
            panel.setCanChooseDirectories(!file);
            panel.setCanCreateDirectories(false);
            panel.setAllowsMultipleSelection(false);
            panel.setResolvesAliases(false);
            panel.setTreatsFilePackagesAsDirectories(false);
            let (prompt, message, suffixes): (_, _, &[&str]) = match selection {
                Selection::GrantFile => (
                    "授权读取",
                    "为此会话选择一个 TXT 或 Markdown 文件",
                    &["txt", "md"],
                ),
                Selection::GrantDirectory => ("授权读取", "为此会话选择一个文件夹", &[]),
                Selection::Textbook => (
                    "导入教材",
                    "选择一个 PDF、DOCX 或 Markdown 教材",
                    &["pdf", "docx", "md"],
                ),
            };
            panel.setPrompt(Some(&NSString::from_str(prompt)));
            panel.setMessage(Some(&NSString::from_str(message)));
            let extensions: Vec<_> = suffixes
                .iter()
                .map(|suffix| NSString::from_str(suffix))
                .collect();
            let content_types = matches!(selection, Selection::Textbook)
                .then(|| {
                    extensions
                        .iter()
                        .map(|extension| UTType::typeWithFilenameExtension(extension))
                        .collect::<Option<Vec<_>>>()
                })
                .flatten();
            if let Some(types) = content_types {
                panel.setAllowedContentTypes(&NSArray::from_retained_slice(&types));
            } else {
                // Preserve P4's extension filter and allow Markdown on systems without a Markdown UTI.
                let extensions = file.then(|| NSArray::from_retained_slice(&extensions));
                #[allow(deprecated)]
                panel.setAllowedFileTypes(extensions.as_deref());
            }

            let selected_panel = panel.clone();
            let reply = Cell::new(Some(reply));
            let completed = RcBlock::new(move |response| {
                let selection = match response {
                    result if result == NSModalResponseOK => selected_panel
                        .URL()
                        .filter(|url| url.isFileURL())
                        .and_then(|url| url.path())
                        .map(|path| Some(path.to_string()))
                        .ok_or(failed),
                    result if result == NSModalResponseCancel => Ok(None),
                    _ => Err(failed),
                };
                if let Some(reply) = reply.take() {
                    let _ = reply.send(selection);
                }
            });
            panel.beginSheetModalForWindow_completionHandler(parent, &completed);
        })
        .map_err(|_| failed)?;
    result.await.map_err(|_| failed)?
}

#[cfg(not(target_os = "macos"))]
pub(crate) async fn pick(
    _window: WebviewWindow,
    _selection: Selection,
    failed: &'static str,
) -> Result<Option<String>, &'static str> {
    Err(failed)
}
