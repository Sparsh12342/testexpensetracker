const path = require('path');
const { BrowserWindow } = require('electron');

const mainWindow = new BrowserWindow({
  width: 800,
  height: 600,
  webPreferences: {
    nodeIntegration: false,  // Usually false for security
    contextIsolation: true,
  }
});

// Assuming main.js is in /electron-app and React build is in ../react-app/dist
mainWindow.loadFile(path.join(__dirname, '../react-app/dist/index.html'));
