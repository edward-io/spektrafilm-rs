//! File actions shared by the canvas and catalog thumbnails.
use egui::Response;
use std::{path::Path, process::Command};

pub(crate) fn show(response: &Response, path: &Path) -> Option<String> {
    let mut error = None;
    response.context_menu(|ui| {
        if ui.button("Copy path").clicked() {
            ui.ctx().copy_text(path.to_string_lossy().into_owned());
            ui.close_menu();
        }
        let label = if cfg!(target_os = "macos") {
            "Show in Finder"
        } else {
            "Show in folder"
        };
        if ui.button(label).clicked() {
            if let Err(message) = reveal(path) {
                error = Some(format!("{label}: {message}"));
            }
            ui.close_menu();
        }
    });
    error
}

fn reveal_command(path: &Path) -> std::io::Result<Command> {
    let path = std::path::absolute(path)?;
    #[cfg(target_os = "macos")]
    let command = {
        let mut command = Command::new("open");
        command.arg("-R").arg(path);
        command
    };
    #[cfg(target_os = "windows")]
    let command = {
        let mut command = Command::new("explorer");
        let mut selection = std::ffi::OsString::from("/select,");
        selection.push(path);
        command.arg(selection);
        command
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let command = {
        let mut command = Command::new("xdg-open");
        command.arg(path.parent().unwrap_or(&path));
        command
    };
    Ok(command)
}

fn reveal(path: &Path) -> Result<(), String> {
    let output = reveal_command(path)
        .and_then(|mut command| command.output())
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        let message = String::from_utf8_lossy(&output.stderr);
        Err(if message.trim().is_empty() {
            output.status.to_string()
        } else {
            message.trim().to_owned()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_click_menu_copies_the_original_path() {
        let ctx = egui::Context::default();
        let path = std::env::temp_dir().join("Photos & exports/日本語 'original'.ARW");
        let frame = |events| {
            ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(500., 400.),
                    )),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        let response = ui.allocate_response(
                            egui::vec2(400., 300.),
                            egui::Sense::click_and_drag(),
                        );
                        assert!(show(&response, &path).is_none());
                        assert!(
                            !response.clicked(),
                            "right-click must not select a thumbnail"
                        );
                    });
                },
            )
        };
        let click = |pos, button| {
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ]
        };
        frame(vec![]);
        frame(click(egui::pos2(50., 50.), egui::PointerButton::Secondary));
        frame(vec![]);
        let menu = frame(vec![]);
        let copy_position = menu
            .shapes
            .iter()
            .find_map(|shape| match &shape.shape {
                egui::epaint::Shape::Text(text) if text.galley.job.text == "Copy path" => {
                    Some(text.pos + text.galley.rect.center().to_vec2())
                }
                _ => None,
            })
            .expect("right-click should open the file menu");
        let copied = frame(click(copy_position, egui::PointerButton::Primary));
        assert!(copied.platform_output.commands.iter().any(|command| {
            matches!(command, egui::OutputCommand::CopyText(text) if text == &path.to_string_lossy())
        }));
    }

    #[test]
    fn reveal_passes_the_literal_path_without_a_shell() {
        let path = std::env::temp_dir().join("Photos & exports/日本語 '$(touch unwanted)'.ARW");
        let command = reveal_command(&path).unwrap();
        let args: Vec<_> = command.get_args().collect();
        #[cfg(target_os = "macos")]
        {
            assert_eq!(command.get_program(), "open");
            assert_eq!(args, [std::ffi::OsStr::new("-R"), path.as_os_str()]);
        }
        #[cfg(target_os = "windows")]
        {
            assert_eq!(command.get_program(), "explorer");
            let mut selection = std::ffi::OsString::from("/select,");
            selection.push(&path);
            assert_eq!(args, [selection.as_os_str()]);
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            assert_eq!(command.get_program(), "xdg-open");
            assert_eq!(args, [path.parent().unwrap().as_os_str()]);
        }
    }
}
