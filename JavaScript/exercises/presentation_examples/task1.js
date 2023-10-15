// -/ Работа с масиви

// --/ 1
const arr = [ 1, 2, 3, 4, 5 ];
const arr1 = [ "a", "b", "c", "d" ];
const arr2 = [ "a", 2, "b", 3 ];

document.write(arr + "<br>")
document.write(arr1 + "<br>")
document.write(arr2 + "<br>")

document.write(arr.length + "<br>")
document.write(arr1.length + "<br>")
document.write(arr2.length + "<br>")

document.write(typeof arr + "<br>")
document.write(typeof arr1 + "<br>")
document.write(typeof arr2 + "<br>")

// --/ 2
let newArr = [];
newArr[0] = 1;
newArr[1] = 2;
newArr[2] = 3;
document.write(newArr + "<br>")
document.write(newArr.length + "<br>")

document.write(newArr.toString() + "<br>")
document.write(newArr.join() + "<br>")
document.write(newArr.pop() + "<br>")
document.write(newArr.push(100) + "<br>")

const mas = [ newArr, [4, 5, 6], [ 7, 8, 9] ];
document.write(mas + "<br>Length of mas is: " + mas.length)
