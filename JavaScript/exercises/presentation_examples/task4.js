// -/ 1
function disableButton() {
    let btn = document.querySelector('#btnDisable');
    if (btn) {
        let disabled = btn.hasAttribute('disabled');
        console.log(disabled);

        btn.setAttribute('name', 'NO!');
        btn.setAttribute('disabled', '');

        disabled = btn.hasAttribute('disabled');
        console.log(disabled);
    }
}

// -/ 2
function attributeOfTag() {
    let link = document.querySelector('#wiki');
    if (link) {
        let target = link.getAttribute('target');
        console.log(target);
    }
}
