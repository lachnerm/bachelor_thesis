/***
    Creates a file that contains the location (x,y,z) and radius of several spheres. Has to be provided with boundary
    sizes where to put the spheres into, the minimum and maximum radius and a padding. The spheres are then randomly
    placed using brute force, such that there will be no intersections.
***/

const pack = require('pack-spheres');
const fs = require('fs')
const parseArgs = require('minimist');

const argv = parseArgs(process.argv.slice(2));
const numberOfMultipliers = argv["multiplier"];
const bounds = argv["bounds"];
const minRadius = argv["minr"];
const maxRadius = argv["maxr"];
const padding = argv["padd"];
const sphereFile = argv["file"];

console.log("Size: ", bounds * 2);
console.log("Minimum radius: ", minRadius);
console.log("Maximum radius: ", maxRadius);
console.log("Padding: ", padding);

const packOptions = {
  dimensions: 3,
  bounds: bounds,
  packAttempts: 500,
  maxCount: 10000,
  minRadius: minRadius,
  maxRadius: maxRadius,
  padding: padding,
};

let allSpheres = Array(numberOfMultipliers).fill().map(() => Array(numberOfMultipliers).fill());
layer_helper_arr = [...Array(numberOfMultipliers).keys()]
layer_helper_arr.forEach(x =>
  layer_helper_arr.forEach(y =>
    allSpheres[x][y] = pack(packOptions)
  )
);

data = "";
let maxZValue = 0;
let xModifier = -(numberOfMultipliers - 1);
allSpheres.forEach((row, rowIndex) => {
  let yModifier = -(numberOfMultipliers - 1);
  row.forEach((cell, colIndex) => {
    cell.forEach(sphere => {
      x = sphere.position[0] + xModifier * bounds;
      y = sphere.position[1] + yModifier * bounds;
      z = sphere.position[2] + bounds;
      radius = sphere.radius;
      data += x + "," + y + "," + z + "," + radius + ",1.52,0\n";

      sphere.position[0] = x;
      sphere.position[1] = y;
      sphere.position[2] = z;
    });
    yModifier += 2;
  });
  xModifier += 2;
});

allSpheres = allSpheres.flat().flat();


xValues = allSpheres.map(sphere => sphere.position[0]);
maxXValue = Math.max(...xValues);
yValues = allSpheres.map(sphere => sphere.position[1]);
maxYValue = Math.max(...yValues);
zValues = allSpheres.map(sphere => sphere.position[2]);
maxZValue = Math.max(...zValues);

fs.writeFile(sphereFile, data, () => {
  console.log("Max X value: %d", maxXValue);
  console.log("Max Y value: %d", maxYValue);
  console.log("Max Z value: %d", maxZValue);
});
