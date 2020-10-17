

require.config({paths: {d3: 'https://d3js.org/d3.v5.min'}});
  
require(["d3"], function(d3) {

    const model_data = {{ model_data }},
    filetype = {{ filetype }},
    filename = {{ filename }},
    save_unfixed = {{ save_unfixed }};

    const width = {{ width }},
        height = {{ height }};

    const nodes = d3.select(".pywr_schematic").selectAll(".node").data();

    // scales to convert back to values between -100 and 100
    const posX = d3.scaleLinear()
                    .range([-100, 100])
                    .domain([0, width]);
    const posY = d3.scaleLinear() 
                   .range([100, -100])
                   .domain([0, height]);

    const output_data = ["Node name,Fixed,x,y"];
    // loop through model nodes and get postions
    for (let i = 0; i < nodes.length; i++){
        
        let node_data = nodes[i];

        if (!(node_data.fixed) && !(save_unfixed)) {
            // If node is unfixed and unfixed node positions are not being saved move to next node
            continue
        }

        if (filetype == "json"){

            let model_node_data = model_data["nodes"].find(node => node.name == node_data.name)

            if ("position" in model_node_data){
                // ensure that any geographic position are not overwritten
                model_node_data["position"]["schematic"] = [posX(node_data.x), posY(node_data.y)]
            } else {
                model_node_data["position"] = {"schematic": [posX(node_data.x), posY(node_data.y)]}
            }
        } else if (filetype == "csv") {
            let position_data = [
                node_data.name,
                node_data.fixed,
                posX(node_data.x),
                posY(node_data.y),
            ];
            output_data.push(position_data.join(","));
        } 
    }

    if (filetype == "json"){
        download(filename, JSON.stringify(model_data));
    } else if (filetype == "csv"){
        download(filename, output_data.join("\n"));
    }
      
}, function(err) {
    element.append("<p style='color:red'>d3.js failed to load:" + err + "</p>");
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
