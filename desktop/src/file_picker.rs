use tauri::WebviewWindow;
use tokio::sync::Mutex;

#[derive(Default)]
pub(crate) struct NativePicker(pub(crate) Mutex<()>);

pub(crate) enum Selection {
    GrantFile,
    GrantDirectory,
    #[cfg(target_os = "macos")]
    Textbook,
    WorkspaceFile,
    SkillDirectory,
    SkillZip,
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
            let file = !matches!(selection, Selection::GrantDirectory | Selection::SkillDirectory);
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
                Selection::WorkspaceFile => (
                    "添加附件",
                    "选择要导入当前会话工作区的文件",
                    &[],
                ),
                Selection::SkillDirectory => (
                    "导入技能文件夹",
                    "选择包含 SKILL.md 的本地技能文件夹",
                    &[],
                ),
                Selection::SkillZip => (
                    "导入技能 ZIP",
                    "选择包含 SKILL.md 的本地 ZIP 压缩包",
                    &["zip"],
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
            } else if !extensions.is_empty() {
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

#[cfg(target_os = "macos")]
pub(crate) async fn pick_save_destination(
    window: WebviewWindow,
    default_filename: String,
    failed: &'static str,
) -> Result<Option<String>, &'static str> {
    use std::cell::Cell;

    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSModalResponseCancel, NSModalResponseOK, NSSavePanel, NSWindow};
    use objc2_foundation::NSString;
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
            let Some(parent) = (unsafe { parent.cast::<NSWindow>().as_ref() }) else {
                let _ = reply.send(Err(failed));
                return;
            };
            let panel = NSSavePanel::savePanel(main_thread);
            panel.setCanCreateDirectories(true);
            panel.setPrompt(Some(&NSString::from_str("另存为")));
            panel.setMessage(Some(&NSString::from_str("选择产物文件的导出位置")));
            panel.setNameFieldStringValue(&NSString::from_str(&default_filename));

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
pub(crate) async fn pick_save_destination(
    _window: WebviewWindow,
    _default_filename: String,
    failed: &'static str,
) -> Result<Option<String>, &'static str> {
    Err(failed)
}

#[cfg(target_os = "macos")]
pub(crate) async fn pick_multiple_workspace_files(
    window: WebviewWindow,
    failed: &'static str,
) -> Result<Vec<String>, &'static str> {
    use std::cell::Cell;

    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSModalResponseCancel, NSModalResponseOK, NSOpenPanel, NSWindow};
    use objc2_foundation::NSString;
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
            let Some(parent) = (unsafe { parent.cast::<NSWindow>().as_ref() }) else {
                let _ = reply.send(Err(failed));
                return;
            };
            let panel = NSOpenPanel::openPanel(main_thread);
            panel.setCanChooseFiles(true);
            panel.setCanChooseDirectories(false);
            panel.setCanCreateDirectories(false);
            panel.setAllowsMultipleSelection(true);
            panel.setResolvesAliases(false);
            panel.setTreatsFilePackagesAsDirectories(false);
            panel.setPrompt(Some(&NSString::from_str("添加附件")));
            panel.setMessage(Some(&NSString::from_str(
                "选择一个或多个文件导入当前会话工作区",
            )));

            let selected_panel = panel.clone();
            let reply = Cell::new(Some(reply));
            let completed = RcBlock::new(move |response| {
                let selection = match response {
                    result if result == NSModalResponseOK => {
                        let urls = selected_panel.URLs();
                        let mut paths = Vec::new();
                        for idx in 0..urls.count() {
                            let url = urls.objectAtIndex(idx);
                            if url.isFileURL() {
                                if let Some(path) = url.path() {
                                    paths.push(path.to_string());
                                }
                            }
                        }
                        Ok(paths)
                    }
                    result if result == NSModalResponseCancel => Ok(Vec::new()),
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
pub(crate) async fn pick_multiple_workspace_files(
    _window: WebviewWindow,
    failed: &'static str,
) -> Result<Vec<String>, &'static str> {
    Err(failed)
}

#[cfg(target_os = "macos")]
pub(crate) async fn pick_multiple_textbooks(
    window: WebviewWindow,
    failed: &'static str,
) -> Result<Vec<String>, &'static str> {
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
            let Some(parent) = (unsafe { parent.cast::<NSWindow>().as_ref() }) else {
                let _ = reply.send(Err(failed));
                return;
            };
            let panel = NSOpenPanel::openPanel(main_thread);
            panel.setCanChooseFiles(true);
            panel.setCanChooseDirectories(false);
            panel.setCanCreateDirectories(false);
            panel.setAllowsMultipleSelection(true);
            panel.setResolvesAliases(false);
            panel.setTreatsFilePackagesAsDirectories(false);
            panel.setPrompt(Some(&NSString::from_str("导入教材")));
            panel.setMessage(Some(&NSString::from_str(
                "选择一个或多个 PDF、DOCX 或 Markdown 教材",
            )));
            let suffixes = ["pdf", "docx", "md"];
            let extensions: Vec<_> = suffixes
                .iter()
                .map(|suffix| NSString::from_str(suffix))
                .collect();
            let content_types = extensions
                .iter()
                .map(|extension| UTType::typeWithFilenameExtension(extension))
                .collect::<Option<Vec<_>>>();
            if let Some(types) = content_types {
                panel.setAllowedContentTypes(&NSArray::from_retained_slice(&types));
            } else {
                let ext_array = NSArray::from_retained_slice(&extensions);
                #[allow(deprecated)]
                panel.setAllowedFileTypes(Some(&ext_array));
            }

            let selected_panel = panel.clone();
            let reply = Cell::new(Some(reply));
            let completed = RcBlock::new(move |response| {
                let selection = match response {
                    result if result == NSModalResponseOK => {
                        let urls = selected_panel.URLs();
                        let mut paths = Vec::new();
                        for idx in 0..urls.count() {
                            let url = urls.objectAtIndex(idx);
                            if url.isFileURL() {
                                if let Some(path) = url.path() {
                                    paths.push(path.to_string());
                                }
                            }
                        }
                        Ok(paths)
                    }
                    result if result == NSModalResponseCancel => Ok(Vec::new()),
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
pub(crate) async fn pick_multiple_textbooks(
    _window: WebviewWindow,
    failed: &'static str,
) -> Result<Vec<String>, &'static str> {
    Err(failed)
}
