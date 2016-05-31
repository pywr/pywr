var graph = {{ graph }};

var div = d3.selectAll(element).append("div");

var width = 500,
    height = 400;

var color = d3.scale.category20();

var force = d3.layout.force()
    .charge(-120)
    .linkDistance(20)
    .size([width, height]);

var svg = div.append("svg")
    .attr("width", width)
    .attr("height", height);

force
  .nodes(graph.nodes)
  .links(graph.links)
  .start();

var link = svg.selectAll(".link")
  .data(graph.links)
.enter().append("line")
  .attr("class", "link")
  .style("stroke", "#333")
  .style("stroke-width", 1);

var node = svg.selectAll(".node")
  .data(graph.nodes)
.enter().append("circle")
  .attr("class", "node")
  .attr("r", 5)
  .style("fill", function(d) { return d.color; })
  .style("stroke", "#333")
  .style("stoke-width", 0.5)
  .call(force.drag);

node.append("title")
  .text(function(d) { return d.name; });

force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

node.attr("cx", function(d) { return d.x; })
    .attr("cy", function(d) { return d.y; });
});
