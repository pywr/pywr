// javascript jinja2 template for drawing a directional graph

// internet explorer can't follow standards so we need some workarounds...
var ua = window.navigator.userAgent;
var browser;
if ((ua.indexOf("MSIE") != -1) || (ua.indexOf("Trident") != -1)) {
    browser = "ie";
    var p = d3.selectAll(element).append("p");
    p.html("Warning: Internet Explorer doesn't support directional arrows on the graph.");
    p.style("color", "#d00");
} else {
    browser = "something better than ie";
}

var graph = {{ graph }};

var style = d3.selectAll(element).append("style");
style.html("{{ css }}");

var div = d3.selectAll(element).append("div");

var width = {{ width }},
    height = {{ height }};

var color = d3.scale.category20();

var force = d3.layout.force()
    .charge(-120)
    .linkDistance(25)
    .size([width, height]);

// workaround for IE not sizing output correctly
div.style("height", height+"px")
   .style("width", width+"px");

var svg = div.append("svg")
    .attr("width", width)
    .attr("height", height);

var posX = d3.scale.linear()
    .range([0, width])
    .domain([-100, 100]);

var posY = d3.scale.linear()
    .range([0, height])
    .domain([100, -100]); // map-style, +ve is up

// set initial node positions 
for (var i = 0; i < graph.nodes.length; i++) {
    var node = graph.nodes[i];
    if (node.position != undefined) {
        node.x = posX(node.position[0]);
        node.y = posY(node.position[1]);
        node.fixed = true;
    } else {
        node.fixed = false;
    }
}

force
  .nodes(graph.nodes)
  .links(graph.links);

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

if(browser == "ie") {
    // internet explorer
    var link = svg.selectAll(".link")
      .data(graph.links)
    .enter().append("line")
      .attr("class", "link")
      .style("stroke", "#333")
      .style("stroke-width", 2);
} else {
    // firefox, etc.
    var link = svg.selectAll(".link")
      .data(graph.links)
    .enter().append("svg:path")
      .attr("class", "link")
      .style("fill", "none")
      .style("stroke", "#333")
      .style("stroke-width", 2)
      .style("marker-end", function(d) { return "url(#end-arrow)"; });
}

var node_size = 5;

function dblclick(d) {
  d3.select(this).classed("fixed", d.fixed = false);
}

function dragstart(d) {
  d3.select(this).classed("fixed", d.fixed = true);
}

var drag = force.drag()
    .on("dragstart", dragstart);

var nodes = svg.selectAll(".node")
              .data(graph.nodes)
              .enter()
              .append("g")  
              .on("dblclick", dblclick)
              .call(drag);

nodes.append("circle")
    .attr("class", "node")
    .attr("r", node_size)
    .attr("class", function(d) {
        var clss = "node";
        for (var i=0; i < d.clss.length; i++) {
            clss += " node-"+d.clss[i];
        };
        return clss;
    });

{% if labels %}
nodes.append("text")
    .attr("dx", 10)
    .attr("dy", 5)
    .style("font-weight", 100)
    .classed("node-text", true)
    .text(function(d){
        return d.name
    });
{% else %}
nodes.append("title")
     .text(function(d) { return d.name; });
{% endif %}

function tick() {
    if(browser == "ie") {
        // internet explorer
        link.attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });
    } else {
        // firefox, etc.
        link.attr("d", function(d) {
            var deltaX = d.target.x - d.source.x,
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

    nodes.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });
}
force.on("tick", tick);

force.start();

{% if attributes %}
    nodes.on("mouseover", function(d){
        
        var table  = d3.selectAll(element)
                    .append("table")
                    .classed("table-tooltip", true);

        var thead = table.append('thead')
        var tbody = table.append('tbody');
        
        var columns = ["attribute", "value"]
        thead.append('tr')
            .selectAll('th')
            .data(columns).enter()
            .append('th')
            .text(function (column) { return column; });
 
        var rows = tbody.selectAll('tr')
                        .data(d["attributes"])
                        .enter()
                        .append('tr');

        var cells = rows.selectAll('td')
                        .data(function (d) {
                            return columns.map(function (column) {
                            return {column: column, value: d[column]};
                            });
                        })
                        .enter()
                        .append('td')
                        .text(function (d) { return d.value; });

    }).on("mouseout", function(){
        d3.select(".table-tooltip").remove()
    });
{% endif %}