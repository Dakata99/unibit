// Примери за изполване на методи 
// за намиране на елемент и модифициране на елементи

// -/ 1
function changeText() {
    let h = confirm("Hello");
    if (h) {
        document.getElementById("pr1").innerText = "Paragraph's text";
    } else {
        alert("Cancel!");
    }
}

// --/ 2 Find element
function findElement() {
    let element = document.getElementById("id1");
    element.innerHTML = "Елементът е намерен!";
}

// --/ 3 Alert a text
function alertText() {
    let text = document.getElementById("id1").innerText;
    alert(text);
}

// --/ 4 Color text
function colorText() {
    const note = document.querySelector(".note");
    note.style.backgroundColor = 'yellow';
    note.style.color = 'red';
}
