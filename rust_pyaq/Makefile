check:
	cargo +nightly check

build:
	cargo +nightly build

release:
	cargo +nightly build --release

run:
	cargo +nighty run --release	-- --self

wasm:
	cargo +nightly build --example rust_pyaq_wasm --target wasm32-unknown-unknown --features="wasm" --release
