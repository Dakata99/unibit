// -/ 1
function createHTMLElement() {
    let h3Element = document.getElementById('hd3');
    let newHeading = document.createElement('h4');
    newHeading = document.createTextNode("Header 4");
    h3Element.appendChild(newHeading);
}

let i = 2;
// -/ 2
function addElementToList() {
    let items = document.querySelector('#items');
    
    let item = document.createElement('li');
    item.textContent = "Item " + (i++).toString();
    items.append(item);
}

// -/ 3
function removeLastElementFromList() {
    let levels = document.getElementById("items");
    levels.removeChild(levels.lastElementChild);
}

// -/ 4
function cloneList() {
    let itemsList = document.querySelector('#items');
    let newItemList = itemsList.cloneNode(true);
    newItemList.id = 'SecondItemList';
    document.body.appendChild(newItemList); // Ще го добави накрая на страницата
}
