const fs = require("fs");

// const inputFile = process.argv[2];
const inputFile = "data/queries.txt";

const inputData = fs.readFileSync(inputFile).toString();

const outputObj = {
    requested: 1000,
    index: "index",
    K: 1.2,
    b: 0.75,
    queries: []
}

const convertText = query => {
    let output = "#combine("
    const terms = query.split(" ");
    terms.forEach((term, idx) => {
        if (term) {
            output += `#bm25(${term})`;
            if (idx < terms.length-1) {
                output += " ";
            }
        }
    });
    output += ")";
    return output;
}

inputData.split(/\r?\n/).forEach(line => {
    const [number, textRaw] = line.split("\t");
    if (number && textRaw) {
        outputObj.queries.push({
            number,
            text: convertText(textRaw)
        });
    }
});

fs.writeFileSync(`${inputFile.split('.')[0]}.json`, JSON.stringify(outputObj, null, 4));