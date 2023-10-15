// -/ 1
function changeStyle() {
    // Change box style
    let box = document.querySelector('.box');
    box.style.backgroundColor = 'yellow';
    box.style.border = 'dashed black';
    box.style.textAlign = 'center';
    // How to center only the box?
    // box.style.p

    // Change paragraph style
    let p = box.querySelector('p');
    p.style.textAlign = 'center';
    p.style.color = 'blue';
}

// -/ 2
function changeBtnColor() {
    let btn = document.getElementById('clrBtn');
    btn.style.color = 'green';
}

// -/ 3
// The same as 2?
