default:
    cargo build --release -p spektrafilm-cli --bin spektrafilm-f64 --features precision-f64
    cargo run --release -p spektrafilm-gui

debug:
    cargo run -p spektrafilm-gui
