function hClick() {
    alert("Натиснахте заглавието!");
}

function modifyText() {

}

function changeText() {
    document.getElementById('p1').innerHTML = 'Change';
}

function collapse() {
    let collapsableLists = document.querySelectorAll('.collapsable');

    for (let i = 0; i < collapsableLists.length; i++) {
        collapsableLists[i].addEventListener('click', function () {
            const content = this.nextElementSibling;
            if (content.style.display == 'block') {
                content.style.display = 'none';
            } else {
                content.style.display == 'block'
            }
        });
    }
}
