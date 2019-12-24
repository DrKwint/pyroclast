function updateImage() {
    const c = document.getElementById("myCanvas");
    const ctx = c.getContext("2d");
    const imgData = ctx.createImageData(128, 128);
    const data = fetch("/sample")
        .then((res) => res.json())
        .then((json) => {
            for (let i = 0; i < imgData.data.length / 4; i += 1) {
                imgData.data[4 * i + 0] = json['sample_values'][3 * i];
                imgData.data[4 * i + 1] = json['sample_values'][3 * i + 1];
                imgData.data[4 * i + 2] = json['sample_values'][3 * i + 2];
                imgData.data[4 * i + 3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
        });

}
