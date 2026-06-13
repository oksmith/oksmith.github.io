---
layout: default
title: Burgess Park Trees
permalink: /burgess-park/
---

<p style="font-size: 13px; color: gray; margin-bottom: 8px;">
  🌳 Click a tree to see if I've taken a photo of it.
</p>

<div style="display: flex; gap: 16px; align-items: flex-start; width: 100vw; position: relative; left: 50%; margin-left: -50vw; padding: 0 16px;">
  <div style="flex: 3;">
    <iframe src="/images/trees/burgess_park_species.html" 
            width="100%" height="850px" frameborder="0" id="tree-map">
    </iframe>
  </div>
  <div id="photo-panel" style="flex: 1; display: none;">
    <img id="tree-photo" style="width: 100%; border-radius: 6px;">
    <p id="tree-label" style="font-size: 13px; color: gray; margin-top: 6px;"></p>
  </div>
</div>

<script>
// Load tree data once on page load
var treeData = {};
fetch('/images/trees/burgess_park_trees.json')
    .then(r => r.json())
    .then(function(trees) {
        trees.forEach(function(t) {
            treeData[t.uniqueid] = t;
        });
    });

window.addEventListener('message', function(e) {
    var uniqueid = e.data.uniqueid;
    if (!uniqueid) return;

    var img = new Image();
    var path = '/images/trees/burgess/' + uniqueid + '.jpg';

    img.onload = function() {
        var tree = treeData[uniqueid] || {};
        document.getElementById('tree-photo').src = path;
        document.getElementById('tree-label').innerHTML = `
            <b>${tree.name_clean || 'Unknown'}</b><br>
            <i>${tree.taxon_species || ''}</i><br>
            Age: ${tree.age_cat || 'Unknown'}<br>
            Girth: ${tree.girth_dbh || 'Unknown'}<br>
            Height: ${tree.height_m || 'Unknown'}<br>
            Canopy: ${tree.canopy_m || 'Unknown'}<br>
            Climate suitability: ${tree.climate_suitability || 'Unknown'}
        `;
        document.getElementById('photo-panel').style.display = 'block';
    };
    img.onerror = function() {
        document.getElementById('photo-panel').style.display = 'none';
    };

    img.src = path;
});
</script>