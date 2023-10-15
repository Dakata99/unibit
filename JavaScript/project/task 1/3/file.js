function addTextToHeaderTags() {
    let idx = 0;
    const headerId = "header";
    let header = null;

    while ((header = document.getElementById(headerId + idx)) != null) {
        header.innerHTML = headerId.toUpperCase() + idx++;
    }
}
