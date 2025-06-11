const sharp = require('sharp');
const fs = require('fs');

async function convertLogo() {
  try {
    // Read SVG file
    const svgBuffer = fs.readFileSync('./public/logo.svg');
    
    // Convert to PNG and save in different sizes
    await sharp(svgBuffer)
      .resize(192, 192)
      .png()
      .toFile('./public/logo192.png');
      
    await sharp(svgBuffer)
      .resize(512, 512)
      .png()
      .toFile('./public/logo512.png');
      
    console.log('Logos converted successfully!');
  } catch (error) {
    console.error('Error converting logos:', error);
  }
}

convertLogo(); 