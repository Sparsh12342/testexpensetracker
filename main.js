const { app, BrowserWindow } = require('electron');
const path = require('path');

const isDev = !app.isPackaged;

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: false,
      nodeIntegration: false,
      webSecurity: !isDev
    }
  });

  if (isDev) {
    win.loadURL('http://localhost:5173').catch((err) => {
      console.error('Failed to load Vite dev server:', err);
      win.loadFile(path.join(__dirname, '../react-app/dist/index.html'));
    });
  } else {
    win.loadFile(path.join(__dirname, '../react-app/dist/index.html'));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
