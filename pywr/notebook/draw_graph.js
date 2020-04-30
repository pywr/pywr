// javascript jinja2 template for drawing a directional graph

require.config({paths: {d3: 'https://d3js.org/d3.v5.min'}});
  
require(["d3"], function(d3) {
    const graph = {{ graph }};

    const links = graph.links.map(d => Object.create(d));
    const nodes = graph.nodes.map(d => Object.create(d));

    const style = d3.selectAll({{ element }}).append("style");
    style.html("{{ css }}");

    const div = d3.selectAll({{ element }}).append("div").classed("pywr_schematic", true);

    const width = {{ width }},
        height = {{ height }};

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(graph.links))
        .force("charge", d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force("x", d3.forceX())
        .force("y", d3.forceY());

    div.style("height", height+"px")
    .style("width", width+"px");

    const svg = div.append("svg")
        .attr("width", width)
        .attr("height", height);

    const posX = d3.scaleLinear()
        .range([0, width])
        .domain([-100, 100]);

    const posY = d3.scaleLinear()
        .range([0, height])
        .domain([100, -100]); // map-style, +ve is up

    // set initial node positions
    for (let i = 0; i < nodes.length; i++) {
        let n = nodes[i];
        if (n.position != undefined) {
            n.fx = posX(n.position[0]);
            n.fy = posY(n.position[1]);
            n.fixed = true;
        } else {
            n.fixed = false;
        }
    }

    // define end-arrow svg marker
    svg.append("svg:defs").append("svg:marker")
    .attr("id", "end-arrow")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 6)
    .attr("markerWidth", 3.5)
    .attr("markerHeight", 3.5)
    .attr("orient", "auto")
    .append("svg:path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#333");

    const link = svg.selectAll(".link")
                    .data(graph.links)
                    .enter().append("svg:path")
                    .attr("class", "link")
                    .style("fill", "none")
                    .style("stroke", "#333")
                    .style("stroke-width", 2)
                    .style("marker-end", function(d) { return "url(#end-arrow)"; });


    function dblclick(d) {
        d.fx = d.x;
        d.fy = d.y;
    }

    function drag() {

        function dragstarted(d) {
        if (!d3.event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        }
        
        function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
        }
        
        function dragended(d) {
        if (!d3.event.active) simulation.alphaTarget(0);
        d.fixed = true;
        }
        
        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }

    const node = svg.selectAll(".node")
                .data(nodes)
                .enter()
                .append("g")  
                .on("dblclick", dblclick)
                .call(drag(simulation));

    let node_size = 5;

    node.append("circle")
        .attr("class", "node")
        .attr("r", node_size)
        .attr("class", function(d) {
            let clss = "node";
            for (let i=0; i < d.clss.length; i++) {
                clss += " node-"+d.clss[i];
            };
            return clss;
        });

    {% if labels %}
    node.append("text")
        .attr("dx", 10)
        .attr("dy", 5)
        .style("font-weight", 100)
        .classed("node-text", true)
        .text(function(d){
            return d.name
        });
    {% else %}
    node.append("title")
        .text(function(d) { return d.name; });
    {% endif %}

    function tick() {

        node.attr("transform", function(d) {
            // ensure nodes do not go beyond svg bounds
            d.x = Math.max(node_size, Math.min(width - node_size, d.x))
            d.y = Math.max(node_size, Math.min(height - node_size, d.y));
            return "translate(" + d.x + "," + d.y + ")";
        });

        link.attr("d", function(d) {
            let deltaX = d.target.x - d.source.x,
                deltaY = d.target.y - d.source.y,
                dist = Math.sqrt(deltaX * deltaX + deltaY * deltaY),
                normX = deltaX / dist,
                normY = deltaY / dist,
                sourcePadding = node_size,
                targetPadding = node_size + 3,
                sourceX = d.source.x + (sourcePadding * normX),
                sourceY = d.source.y + (sourcePadding * normY),
                targetX = d.target.x - (targetPadding * normX),
                targetY = d.target.y - (targetPadding * normY);
            return "M" + sourceX + "," + sourceY + "L" + targetX + "," + targetY;
        });
    }
    simulation.on("tick", tick);

    {% if attributes %}
        node.on("mouseover", function(d){
            
            d3.select(".table-tooltip").remove();

            const table  = d3.selectAll({{ element }})
                        .append("table")
                        .classed("table-tooltip", true);

            const thead = table.append('thead')
            const tbody = table.append('tbody');
            
            const columns = ["attribute", "value"]
            thead.append('tr')
                .selectAll('th')
                .data(columns).enter()
                .append('th')
                .text(function (column) { return column; });

            const table_data = Object.assign([], d["attributes"])
            table_data.push({"attribute": "x coordinate", "value": d.x.toFixed(2)})
            table_data.push({"attribute": "y coordinate", "value": d.y.toFixed(2)})

            const rows = tbody.selectAll('tr')
                            .data(table_data)
                            .enter()
                            .append('tr');
            
            rows.selectAll('td')
                .data(function (d) {
                    return columns.map(function (column) {
                    return {column: column, value: d[column]};
                    });
                })
                .enter()
                .append('td')
                .text(function (d) { return d.value; });

        }).on("mouseout", function(){
            d3.select(".table-tooltip").transition().delay(2000).remove();
        });
    {% endif %}

}, function(err) {
    element.append("<p style='color:red'>d3 failed to load:" + err + "</p>");   
});
