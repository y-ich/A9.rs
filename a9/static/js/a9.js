/* global WebDNN Rust */

let network = null;

async function evaluate(feature) {
    const views = network.getInputViews();
    views[0].set(feature);
    try {
        await network.run();
    } catch (e) {
        console.log(e);
        return null;
    }
    const output = network.getOutputViews()[0].toActual();
    return output;
}

const networkPromise = WebDNN.load('./output', {
    backendOrder: ['webgpu']
}).catch(function(e) {
    console.log(e);
});


Promise.all([networkPromise, Rust.liba9]).then(function(data) {
    network = data[0];
    console.log(network);
    const liba9 = data[1];
    console.log(liba9.think([], 100));
});
