/**
 * generate_icons.js
 * 
 * Run this ONCE in Node.js to generate the PNG icon files
 * needed by the Chrome extension.
 * 
 * Usage: node generate_icons.js
 * 
 * Requires: npm install canvas
 * (or use the pre-built SVG approach below if canvas isn't available)
 */

const fs = require("fs");
const path = require("path");

// Try to use the 'canvas' package if available
let canvasAvailable = false;
try {
    const { createCanvas } = require("canvas");
    canvasAvailable = true;

    const sizes = [16, 48, 128];
    const iconsDir = path.join(__dirname, "icons");
    if (!fs.existsSync(iconsDir)) fs.mkdirSync(iconsDir);

    sizes.forEach(size => {
        const canvas = createCanvas(size, size);
        const ctx = canvas.getContext("2d");

        // Background gradient
        const grad = ctx.createLinearGradient(0, 0, size, size);
        grad.addColorStop(0, "#4f46e5");
        grad.addColorStop(0.6, "#6366f1");
        grad.addColorStop(1, "#818cf8");
        ctx.fillStyle = grad;

        // Rounded rect
        const r = size * 0.2;
        ctx.beginPath();
        ctx.moveTo(r, 0);
        ctx.lineTo(size - r, 0);
        ctx.arcTo(size, 0, size, r, r);
        ctx.lineTo(size, size - r);
        ctx.arcTo(size, size, size - r, size, r);
        ctx.lineTo(r, size);
        ctx.arcTo(0, size, 0, size - r, r);
        ctx.lineTo(0, r);
        ctx.arcTo(0, 0, r, 0, r);
        ctx.closePath();
        ctx.fill();

        // Ticket icon (simplified)
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        const s = size * 0.55;
        const x = (size - s) / 2;
        const y = (size - s) / 2;
        ctx.fillRect(x, y, s, s * 0.1);          // top bar
        ctx.fillRect(x, y + s * 0.3, s, s * 0.1); // middle bar
        ctx.fillRect(x, y + s * 0.6, s * 0.6, s * 0.1); // short bar

        // Lightning bolt
        const bx = size * 0.58;
        const by = size * 0.55;
        ctx.fillStyle = "rgba(255,230,50,0.95)";
        ctx.beginPath();
        ctx.moveTo(bx + size * 0.08, by);
        ctx.lineTo(bx, by + size * 0.1);
        ctx.lineTo(bx + size * 0.05, by + size * 0.1);
        ctx.lineTo(bx - size * 0.01, by + size * 0.2);
        ctx.lineTo(bx + size * 0.05, by + size * 0.11);
        ctx.lineTo(bx, by + size * 0.11);
        ctx.closePath();
        ctx.fill();

        const buffer = canvas.toBuffer("image/png");
        const outPath = path.join(iconsDir, `icon${size}.png`);
        fs.writeFileSync(outPath, buffer);
        console.log(`✅ Generated ${outPath}`);
    });

} catch (e) {
    // canvas not available — create SVG-as-PNG placeholder
    console.log("ℹ️  canvas package not found. Creating SVG placeholders...");
    console.log("   Install with: npm install canvas");
    console.log("   Then re-run this script to get proper PNG icons.\n");

    const iconsDir = path.join(__dirname, "icons");
    if (!fs.existsSync(iconsDir)) fs.mkdirSync(iconsDir);

    const svgContent = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#4f46e5"/>
      <stop offset="60%"  stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#818cf8"/>
    </linearGradient>
  </defs>
  <rect width="128" height="128" rx="26" fill="url(#g)"/>
  <rect x="24" y="36" width="56" height="8"  rx="3" fill="white" opacity="0.95"/>
  <rect x="24" y="60" width="56" height="8"  rx="3" fill="white" opacity="0.95"/>
  <rect x="24" y="84" width="36" height="8"  rx="3" fill="white" opacity="0.95"/>
  <polygon points="84,58 74,78 80,78 72,98 92,72 84,72 93,58" fill="#fde047" opacity="0.95"/>
</svg>`;

    // Write the SVG as the icon (Chrome accepts SVG in some contexts but not action icons)
    // So write it as a note file
    fs.writeFileSync(path.join(iconsDir, "icon.svg"), svgContent);
    console.log("✅ Created icons/icon.svg");

    // Create minimal 1x1 placeholder PNGs so the extension loads without errors
    // Real PNG header + 16x16 solid indigo
    const minimalPng16 = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAKUlEQVQ4T2NkYGBg" +
        "YGBgQEMMMDAqwAAAAABJRU5ErkJggg==",
        "base64"
    );
    const minimalPng48 = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAALElEQVRo3u3BAQ0A" +
        "AAwCoPVP7WsIoAAAAAAAAAAAAAAAAAAAAAAAeAMBxAABIgBJ6QAAAABJRU5ErkJggg==",
        "base64"
    );
    const minimalPng128 = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAIAAAACAAAAAAeKHoUAAAAM0lEQVR42u3BAQ0AAAAC" +
        "IOP+3dEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AQAAK0AAAABJRkJggg==",
        "base64"
    );

    fs.writeFileSync(path.join(iconsDir, "icon16.png"), minimalPng16);
    fs.writeFileSync(path.join(iconsDir, "icon48.png"), minimalPng48);
    fs.writeFileSync(path.join(iconsDir, "icon128.png"), minimalPng128);

    console.log("⚠️  Placeholder PNGs written. For proper icons:");
    console.log("   1. npm install canvas");
    console.log("   2. node generate_icons.js");
    console.log("   Or manually place your own 16x16, 48x48, 128x128 PNGs in the icons/ folder.");
}

console.log("\n🎉 Done! Icons ready in: " + path.join(__dirname, "icons"));
