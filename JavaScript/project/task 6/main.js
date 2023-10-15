class Task {
    constructor(topic, description = "", status, priority) {
        this.topic = topic;
        this.description = description;
        this.status = status;
        this.priority = priority;
    }
}

let colors = {
    // Status
    "Open": "gray",
    "In progress": "blue",
    "Done": "green",
    // Priorities
    "Low": "yellow",
    "Medium": "orange",
    "High": "red",
};

// Helper container to track the tasks easier
let tasks = [];

function createTask() {
    let topic = document.getElementById("topic").value;
    let description = document.getElementById("description").value;
    let status = document.getElementById("status").value;
    let priority = document.getElementById("priority").value;

    if (status == "Done" || status == "In progress") {
        alert("Can not create a task into 'In progress' or 'Done' status!");
        return;
    }

    // Create new task
    let task = new Task(topic, description, status, priority);

    let ulist = null;
    // Create an unordered list if there are no tasks
    if (tasks.length == 0) {
        ulist = document.createElement("ul");
        ulist.id = "to-do-list";
    } else { // otherwise get that list
        ulist = document.getElementById("to-do-list");
    }

    // Alert if the topic is already added
    if (findElementIndex(topic) != -1) {
        alert("Task already added!");
    } else { // otherwise add it to the list
        tasks.push(task);
        let li = document.createElement("li");

        let p1 = document.createElement('p');
        p1.textContent = "Topic: " + task.topic;

        let p2 = document.createElement('p');
        p2.textContent = "Description: ";

        let span = document.createElement('span');
        span.textContent = task.description;
        p2.appendChild(span);

        let p3 = document.createElement('p');
        p3.textContent = "Status: ";

        span = document.createElement('span');
        span.textContent = task.status;
        span.style.backgroundColor = colors[task.status];
        p3.appendChild(span);

        let p4 = document.createElement('p');
        p4.textContent = "Priority: ";

        span = document.createElement('span');
        span.textContent = task.priority;
        span.style.backgroundColor = colors[task.priority];
        p4.appendChild(span);

        li.appendChild(p1);
        li.appendChild(p2);
        li.appendChild(p3);
        li.appendChild(p4);

        li.id = topic; // Topics are unique
        ulist.appendChild(li);
    }

    let listContainer = document.getElementById("list-container");
    listContainer.appendChild(ulist);
}

function modifyTask() {
    let topic = document.getElementById("topic").value;
    let description = document.getElementById("description").value;
    let status = document.getElementById("status").value;
    let priority = document.getElementById("priority").value;

    // Find the task by topic (it is unique)
    let index = findElementIndex(topic);

    if (index == -1) {
        alert("There is no such task to modify!");
    } else {
        let li = document.getElementById(topic);

        // If user wants to delete the task, which is in Done status
        if (status == "Done" && confirm("Do you want to delete the task from the list?")) {
            let ulist = li.parentNode;
            ulist.removeChild(li);
            tasks.splice(index, 1);

            // If there are no tasks left, delete the unordered list
            if (tasks.length == 0) {
                let ulist = document.getElementById("to-do-list");
                if (ulist) {
                    let parent = ulist.parentNode;
                    parent.removeChild(ulist);
                }
            }
            return;
        }

        // Modify the description
        let p = li.querySelector('p:nth-child(2)'); // 2nd paragraph
        let span = p.querySelector('span');
        span.innerHTML = description;

        // Modify the status
        p = li.querySelector('p:nth-child(3)'); // 3rd paragraph
        span = p.querySelector('span');
        span.innerHTML = status;
        span.style.backgroundColor = colors[status];

        // Modify the prority
        p = li.querySelector('p:nth-child(4)');  // 4th paragraph
        span = p.querySelector('span');
        span.innerHTML = priority;
        span.style.backgroundColor = colors[priority];
    }
}

function findElementIndex(topic) {
    for (let i = 0; i < tasks.length; i++) {
        if (tasks[i].topic == topic) {
            return i;
        }
    }
    return -1;
}

// Enable buttons when an input is provided for topic (its required)
function checkInput() {
    let topic = document.getElementById("topic");
    let createBtn = document.getElementById("createBtn");
    let modifyBtn = document.getElementById("modifyBtn");

    topic.addEventListener('input', function () {
        if (topic.value.trim() != '') {
            createBtn.disabled = false; // Enable the button
            modifyBtn.disabled = false; // Enable the button
        } else {
            createBtn.disabled = true; // Disable the button
            modifyBtn.disabled = true; // Disable the button
        }
    })
}
checkInput();

// DOM Methods used:
// getElementById, querySelector, createElement, appendChild, removeChild, addEventListener

// DOM Properties used:
// document, parentNode, innerHTML, textContent, value, style, id
