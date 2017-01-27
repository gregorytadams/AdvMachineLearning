//Influence taken from blog posts by Mike Bostock at https://bl.ocks.org/mbostock/4062045

$(document).ready(function(){

  var width = 1140,
      height = 625;
  
  var color = d3.scale.ordinal()
      .domain(['r', 'd', 'o'])
      .range(['#FF563B','#29859F','#FF9E3B']);
  
  var force = d3.layout.force()
      .charge(-500)
      .linkDistance(function(d){ return (d.weight) + (Math.log((1 / d.weight)) * -50); })
      .size([width, height]);
  
  //Creates box to plot in based on width and height
  var svg = d3.select("div #box").append("svg:svg")
      .attr("width", width)
      .attr("height", height)
  
    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .style('fill', 'none')
      .style('stroke', '#000')
      .style('stroke-width', 10);

  function add_nodes_and_edges(file_name) {
      svg.selectAll('.node')
          .remove()
      svg.selectAll('.link')
          .remove()

    d3.json(file_name, function(error, graph) {
      if (error) throw error;

      force
          .nodes(graph.nodes)
          .links(graph.links)
          .start();

      var link = svg.selectAll(".link")
          .data(graph.links)
          .enter().append("line")
          .attr("class", "link")
          .style("stroke-width", function(d) { return d.weight / 5; })
          .style("stroke", '#C2C2C2');
      
      link.append("title")
          .text(function(d) {return d.weight})     
      var node = svg.selectAll(".node")
          .data(graph.nodes)
          .enter().append("circle")
          .attr("class", "node")
          .attr("r", function(d){ return (d.size / 500) + 5})
          .style("fill", function(d) { return color(d.party); })
          .call(force.drag)
          .on('dblclick', function(d, i) {
            url = 'senators/' + d.id + '/';
            window.location.href = url;
          })
          .on('mouseover', function(d) {
              node.transition()
                  .duration(300)
                  .style("r", function(l) {
                      if (d === l)
                        return (d.size / 250) + 5;
                      else
                        return (l.size / 1500) + 5;  
                      })

              link.transition()
                  .duration(300)
                  .style('stroke', function(l) {
                  if (d === l.target)
                    return color(l.source.party);
                  else if (d === l.source)
                    return color(l.target.party);
                  else
                    return '#C2C2C2';
                  })
                  .style('stroke-width', function(l) {
                  if (d === l.source || d === l.target)
                    return l.weight / 2;
                  else
                    return l.weight / 10;
                  });
          })
          .on('mouseout', function(d) {
              node.transition()
                .duration(300)
                .style('r', function(l){ return (l.size / 500) + 5})
              link.transition()
                .duration(300)
                .style('stroke', '#C2C2C2')
                .style('stroke-width', function(l){
                  return l.weight / 5;
                });
          });
      node.append("title")
            .text(function(d) { return d.label; })
      force.on("tick", function() {
        node.attr("cx", function(d) { return d.x = Math.max(15, Math.min(width - 15, d.x)); })
            .attr("cy", function(d) { return d.y = Math.max(15, Math.min(height - 15, d.y)); });
    
        link.attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });    
      });
    });
  }
  
  add_nodes_and_edges('/static/gov_data/force_json/114_force.json');

  $("body").on("click", "#senate", (function () {
    var value = $("select").find('option:selected').val();
    add_nodes_and_edges('/static/gov_data/force_json/' + value + '_force.json')
  }));

  $("body").on("click", "#democrat", (function () {
    var value = $("select").find('option:selected').val(); 
    add_nodes_and_edges('/static/gov_data/party_json/' + value + '_dem_force.json')
  }));  

  $("body").on("click", "#republican", (function () {
    var value = $("select").find('option:selected').val();
    add_nodes_and_edges("/static/gov_data/party_json/" + value + "_rep_force.json")
  }));

  $('select').on('change', function() {
      add_nodes_and_edges('/static/gov_data/force_json/' + this.value + '_force.json')
  });
});