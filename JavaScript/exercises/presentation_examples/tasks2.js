// -/ 2
function changeButton() {
    let btn = document.getElementById('btnChng');
    btn.innerHTML = 'Натиснахте бутона!';
}

// -/ 3
function changeText1() {
    let elem = document.getElementById('pr2');
    elem.innerText = 'Фокус';
}

function changeText2() {
    let elem = document.getElementById('pr2');
    elem.innerText = 'Вече не сте във фокус';
}

// -/ 4
function fullLoad() {
    if (document.readyState == 'complete') {
        alert('Страницата е заредена успешно');
    }
}
fullLoad();
