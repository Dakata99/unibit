// -/ 1
function createObject1() {
    let student = {
        firstName: "Daniel",
        lastName: "Lyubenov",
        age: 23,
        grade: "2",
        nomer: 192
    };

    document.write(student.firstName + "<br>");
    document.write(student.lastName + "<br>");
    document.write(student.age + "<br>");
    document.write(student.grade + "<br>");
    document.write(student.nomer + "<br>");
}
createObject1();

// -/ 2
function createObject2() {
    let student = new Object();
    student.firstName = "Daniel";
    student.lastName = "Lyubenov";
    student.age = 23;
    student.height = 185;

    document.write(student.firstName + ' ' + student.lastName + ", " + student.age + " години, ръст " +
                    student.height + "<br>");
}
createObject2();

// -/ 3
function Student(firstName, lastName, age, grade, nomer) {
    this.firstName = firstName;
    this.lastName = lastName;
    this.age = age;
    this.grade = grade;
    this.nomer = nomer;
}

function printStundents() {
    let students = new Array();
    students[0] = new Student("Ivan", "Ivanov", 22, "12a", 23);
    students[1] = new Student("Ivana", "Ivanova", 13, "4a", 5);
    students[2] = new Student("Gero", "Gerasimov", 15, "8a", 9);
    students[3] = new Student("Pesho", "Peshov", 11, "11a", 11);

    for (st in students) {
        document.write(students[st].firstName + ' ' + students[st].lastName + ", " + students[st].age + 
        " години, клас " + students[st].grade + ", номер " + students[st].nomer + "<br>");
    }
}
printStundents();

// -/ 4
function t4() {
    const student = {
        firstName: "Daniel",
        lastName: "Lyubenov",
        age: 23,
        eyeColor: "brown"
    };
    document.getElementById('t4').innerHTML = student.firstName + " is " + student.age + " years old.";
}
t4();

// -/ 5
function t5() {
    const student = {
        firstName: "Daniel",
        lastName: "Lyubenov",
        age: 23,
        printing() {
            console.log("Name: ", this.firstName);
        }
    };

    let st = Object.create(student);
    st.firstName = "Rambo";
    st.printing();
}
t5();

// -/ HTML Forms
function focusBlur() {
    document.querySelector("input").focus();
    document.querySelector("input").blur();
}
focusBlur();

function getLength() {
    let text = document.getElementById("inputLength");
    let output = document.querySelector("#length");
    text.addEventListener("input", function () {
        output.textContent = text.value.length;
    });
}
getLength();

function save() {
    let form = document.getElementById("f2");
    form.addEventListener("submit", function (event) {
        console.log("Saving value", form.elements.value.value);
        event.preventDefault();
    });
}
save();

function changeFormColor() {
    let checkbox = document.querySelector("#red");
    checkbox.addEventListener("change", function () {
        document.body.style.background = checkbox.checked ? "red" : "";
    });
}
changeFormColor();

function radioBtnColor() {
    let buttons = document.getElementsByName("color");
    for (let i = 0; i < buttons.length; i++) {
        buttons[i].addEventListener("change", function () {
            document.body.style.background = buttons[i].value;
        })
    }
}
radioBtnColor();

function binaryString() {
    let select = document.querySelector("select");
    let output = document.querySelector("#output");

    select.addEventListener("change", function () {
        let sum = 0;
        for (let i = 0; i < select.options.length; i++) {
            let option = select.options[i];
            if (option.selected) {
                sum += Number(option.value);
            }
        }
        output.textContent = sum;
    });
}
binaryString();

function fileInput() {
    let fileInput = document.getElementById("fileInput");
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            let file = fileInput.files[0];
            console.log("File: ", file.name);
            console.log("Type: ", file.type); // Should be exception proof
        }
    });
}
fileInput();

function addWord() {
    let text1 = document.form1.text1.value;
    if (text1 == "") {
        windows.alert('Enter text');
        return "";
    }
    let ta1 = document.form1.ta1.value;
    let result = ta1 + text1 + "\n";
    document.form1.ta1.value = result;
    document.form1.text1.value = "";
    return text1;
}

function validateEmail(email) {
    let regex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
    return regex.test(email);
}

function check() {
    let email = document.form2.email.value;
    if (validateEmail(email)) {
        alert("Email is OK.");
        document.getElementById("span1").innerHTML = "Email is OK.";
    } else {
        alert("Email is not OK!");
        document.getElementById("span1").innerHTML = "Email is not OK!";
    }
}

