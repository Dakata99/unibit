let hobbies_eng = [
    { key: "Travelling", value: "../images/travel.jpg" },
    { key: "Gaming", value: "../images/gaming.jpg" },
    { key: "Fitness", value: "../images/fitness.png" },
    { key: "Music", value: "../images/music.jpg" }
];

let hobbies_bg = [
    { key: "Пътуване", value: "../images/travel.jpg" },
    { key: "Игри", value: "../images/gaming.jpg" },
    { key: "Фитнес", value: "../images/fitness.png" },
    { key: "Музика", value: "../images/music.jpg" }
];

// Set the proper gallery according to the language set from the <html> tag
let hobbies = document.documentElement.lang == "en" ? hobbies_eng : hobbies_bg;

let idx = 0;

function slideshow() {
    let img = document.getElementById("slide-img");

    // Start over
    if (idx > hobbies.length - 1) {
        idx = 0;
    }

    // Change the image caption
    let imgcaption = document.getElementById("imgcaption");
    imgcaption.innerHTML = hobbies[idx].key;

    // Set the image source
    img.alt = hobbies[idx].key;
    img.src = hobbies[idx++].value;

    // Change the image
    setTimeout(slideshow, 2500);
}
slideshow();
