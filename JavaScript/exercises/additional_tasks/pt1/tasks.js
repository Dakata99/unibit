// -/ 1
function clearValue1() {
    document.getElementById('number1').value = '';
    document.getElementById('result').value = '';
}

function clearValue2() {
    document.getElementById('number2').value = '';
    document.getElementById('result').value = '';
}

function sum() {
    let number1 = document.getElementById('number1').value;
    let number2 = document.getElementById('number2').value;
    document.getElementById('result').value = parseInt(number1) + parseInt(number2);
}

// -/ 2
function checkAnswer() {
    let version = document.querySelector('input[name="html_version"]:checked').value;

    if (version == "true") {
        alert("Correct!");
    } else {
        alert("Not correct!");
    }
}

// -/ 3
function checkSelectedOptions() {
    const selectElement = document.getElementById('options');
    const selectedOptions = [];

    for (let i = 0; i < selectElement.options.length; i++) {
        if (selectElement.options[i].selected) {
            selectedOptions.push(selectElement.options[i].value);
        }
    }

    console.log(selectedOptions);
}

// -/ 4
function checkCheckboxes() {
    let boxes = document.getElementsByName('checkbox');
    let selectedBoxes = [];

    for (let i = 0; i < boxes.length; i++) {
        if (boxes[i].checked) {
            selectedBoxes.push(boxes[i].value);
        }
    }
    alert("Вашите избори са: " + selectedBoxes);
}

// -/ 5
function checkFileUpload() {
    const fileInput = document.getElementById("fileInput");

    if (fileInput.files.length > 0) {
        console.log("Файл " + fileInput.files[0].name + " е успешно качен.");
    } else {
        console.log("Няма избран файл за качване!");
    }
}