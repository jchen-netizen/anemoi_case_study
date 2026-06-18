// Run in Google Earth Engine Code Editor

// var geometry = /* color: #d63000 */ee.Geometry.MultiPoint(),
geometry2 = /* color: #98ff00 */ee.Geometry.Point([-121.83928, 57.20271]);

// Define a region of interest and center the map
var roi = geometry2; // Donnie Creek
Map.centerObject(roi, 9);
var region = roi.buffer(1000000) // 1000 km radius around the point

// Helper: list of dates from start to end (inclusive)
var startDate = ee.Date('2023-05-01');
var endDate = ee.Date('2023-06-20');
var nDays = endDate.difference(startDate, 'day').getInfo(); // 48 days

var tempVis = {
  min: -5,
  max: 45,
  palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
};

var fireVis = {
  bands: ['T21'],
  min: 300,
  max: 400,
  palette: ['red']
};

// Loop over each day
for (var i = 0; i < nDays; i++) {
  
  var dayStart = ee.Date('2023-05-01').advance(i, 'day');
  var dayEnd = ee.Date('2023-05-01').advance(i + 1, 'day');
  
  var dateStr = dayStart.format('YYYY-MM-dd').getInfo();
  
  // Temperature: MODIS Terra Daily LST Collection (Version 6.1)
  var temp = ee.ImageCollection('MODIS/061/MOD11A1')
                  .filterDate(dayStart, dayEnd);
  
  var lstDay = temp.select('LST_Day_1km');
  
  var lstCelsius = lstDay.map(function(image) {
  return image.multiply(0.02).subtract(273.15)
              .copyProperties(image, ['system:time_start']);
  });
  
  var lstVisualized = lstCelsius.mean();
  
  Map.addLayer(lstCelsius.mean(), tempVis, 'Mean Daytime LST (Celsius)');
  
  Export.image.toDrive({
  image: lstVisualized,
  description: 'meanDaytimeLST_colorized_' + dateStr,
  folder: 'GEE_LST_Images',   
  region: region,         
  scale: 1000,
  maxPixels: 1e9
  });
  
  // FIRMS: derived from the same MODIS Terra/Aqua as Fires and Thermal Anomalies
  var firms = ee.ImageCollection('FIRMS')
    .filterDate(dayStart, dayEnd);

  // var fireVis = {      // not "science quality" for quantitative analysis
  //   bands: ['T21'],
  //   min: 300,
  //   max: 400,
  //   palette: ['yellow', 'orange', 'red']
  // };

  Map.addLayer(firms, fireVis, 'Fires and Thermal Anomalies');

  var fireVisualized = firms.select('T21').max();

  Export.image.toDrive({
    image: fireVisualized,
    description: 'Fire_and_Thermal_Anomalies_max_' + dateStr,
    folder: 'GEE_LST_Images',   
    region: region,         
    scale: 1000,
    maxPixels: 1e9
  });

}


