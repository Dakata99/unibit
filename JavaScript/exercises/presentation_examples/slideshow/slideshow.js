let num = 0;
let images = new Array(
    [ '1.png', 'Picture 1' ],
    [ '2.png', 'Picture 2' ]
);

function slideshow(slide_num) {
    document.getElementById('mypic').src = images[slide_num][0];
    document.getElementById('mypic').alt = images[slide_num][1];
    document.getElementById('descr').innerHTML = images[slide_num][1];

    let img = document.getElementById('mypic');
    img.style.height = "400px";
    img.style.width = "auto";

    let size = images.length;
    if (slide_num == 0 ) {
        document.getElementById("Prev").disabled = true;
    } else {
        document.getElementById("Prev").disabled = false;
    }

    if (slide_num == (size - 1)) {
        document.getElementById("Next").disabled = true;
    } else {
        document.getElementById("Next").disabled = false;
    }
}

function slideshowUp() {
    num++;
    slideshow(num);
}

function slideshowBack() {
    num--;
    slideshow(num);
}