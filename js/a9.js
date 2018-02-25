/* global WebDNN WebAssembly */

let network = null;

async function evaluate(feature) {
    const views = network.getInputViews();
    views[0].set(feature);
    await network.run();
    const output = network.getOutputViews()[0].toActual();
    return output;
}

const networkPromise = WebDNN.load('./output', {
    backendOrder: ['webgpu']
}).catch(function(e) {
    console.log(e);
});


const wasmPromise = fetch('./js/rust_pyaq_wasm.wasm')
    .then(response => response.arrayBuffer())
    .catch(e => console.log(e))
    .then(bytes => WebAssembly.instantiate(bytes, {
        imported_func: function(arg) {
            console.log(arg);
        }
    }))
    .catch(e => console.log(e));

Promise.all([networkPromise, wasmPromise]).then(function(data) {
    network = data[0];
    const wasm = data[1];
    console.log(wasm.instance.think());
});
