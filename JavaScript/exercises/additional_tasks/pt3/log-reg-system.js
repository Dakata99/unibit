class User {
    constructor(name, lname, email, uname, passwd) {
        this.name = name;
        this.lname = lname;
        this.email = email;
        this.uname = uname;
        this.passwd = passwd;
    }
    printInfo() {
        let info = "Name: " + this.name + "\n" + 
                   "Last name: " + this.lname + "\n" +
                   "Email: " + this.email + "\n" +
                   "Username: " + this.uname + "\n" +
                   "Password: " + this.passwd + "\n";
        console.log(info);
    }
}

// TODO: add a check if user is already registered
// (not hard to do, but will skip it)
let users = new Array();

function checkInputs(arr) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] == '' || arr[i] == null) {
            return false;
        }
    }
    return true;
}

function clearInputs1() {
    document.getElementById("name").value = "";
    document.getElementById("lname").value = "";
    document.getElementById("email").value = "";
    document.getElementById("uname").value = "";
    document.getElementById("passwd").value = "";
}

function clearInputs2() {
    document.getElementById("logUName").value = "";
    document.getElementById("logPasswd").value = "";
}

function registration() {
    let name = document.getElementById("name").value;
    let lname = document.getElementById("lname").value;
    let email = document.getElementById("email").value;
    let uname = document.getElementById("uname").value;
    let passwd = document.getElementById("passwd").value;

    if (checkInputs(Array(name, lname, email, uname, passwd)) == false) {
        alert("Please fulfill all boxes!");
        document.getElementById("regStatus").innerHTML = "Registration failed!";
        setTimeout(function () {
            document.getElementById("regStatus").innerHTML = "";
        }, 900);
        return;
    }

    let newUser = new User(name, lname, email, uname, passwd);

    users.push(newUser);
    console.log(users);
   
    document.getElementById("regStatus").innerHTML = "Successfully registrated!";
    clearInputs1();
    setTimeout(function () {
        document.getElementById("regStatus").innerHTML = "";
    }, 900);
}

function login() {
    let uname = document.getElementById("logUName").value;
    let passwd = document.getElementById("logPasswd").value;

    if (checkInputs(Array(uname, passwd)) == false) {
        alert("Please fulfill all boxes!");
        document.getElementById("logStatus").innerHTML = "Login failed!";
        setTimeout(function () {
            document.getElementById("logStatus").innerHTML = "";
        }, 900);
        return;
    }

    let flag = false;
    for (let i = 0; i < users.length; i++) {
        if (users[i].uname == uname && users[i].passwd == passwd) {
            document.getElementById("logStatus").innerHTML = "Successfully logged in!";
            flag = true;
        }
    }

    if (!flag) {
        alert("Invalid username or password!");
    }

    setTimeout(function () {
        document.getElementById("logStatus").innerHTML = "";
    }, 900);
    clearInputs2();
}