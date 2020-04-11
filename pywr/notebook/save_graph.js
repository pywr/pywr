

require.config({paths: {d3: 'https://d3js.org/d3.v5.min'}});
  
require(["d3"], function(d3) {

    const model_data = {{ model_data }};

    const width = {{ width }},
        height = {{ height }};

    const nodes = d3.select(".pywr_schematic").selectAll(".node").data();

    const node_data_obj = {}
    nodes.forEach(element => {
        node_data_obj[element.name] = element
    });

    // scales to convert back to values between -100 and 100
    const posX = d3.scaleLinear()
                    .range([-100, 100])
                    .domain([0, width]);
    const posY = d3.scaleLinear() 
                   .range([100, -100])
                   .domain([0, height]);
    
    for (let i = 0; i < model_data.nodes.length; i++){
        let node_name = model_data.nodes[i].name;
        let node_data = node_data_obj[node_name]
        if (node_data.fixed){
            model_data["nodes"][i]["position"] = {"schematic": [posX(node_data.x), posY(node_data.y)]}
        }
    }

    const filename = {{ filename }}
    download(filename + ".json", JSON.stringify(model_data))   
});

function download(filename, text) {
    let pom = document.createElement('a');
    pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    pom.setAttribute('download', filename);

    if (document.createEvent) {
        let event = document.createEvent('MouseEvents');
        event.initEvent('click', true, true);
        pom.dispatchEvent(event);
    }
    else {
        pom.click();
    }
}

