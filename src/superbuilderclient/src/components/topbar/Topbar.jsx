import './Topbar.css';
import React, { useEffect, useState, useContext } from 'react';
import { getAllWindows, getCurrentWindow } from '@tauri-apps/api/window';
import AssistantLogo from '../assistantLogo/assistantLogo';
import { AppStatusContext } from '../context/AppStatusContext';
import { useTranslation } from 'react-i18next';
import { IconButton } from '@mui/material';
import MinimizeIcon from '@mui/icons-material/Minimize';
import CloseIcon from '@mui/icons-material/Close';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import CloseFullscreenIcon from '@mui/icons-material/CloseFullscreen';
import useDataStore from '../../stores/DataStore';
const appWindow = getCurrentWindow();

const Topbar = ({ children }) => {
  const { t } = useTranslation();
  const { assistant, assistantName } = useDataStore();
  const [isMaximized, setMaximized] = useState(false);

  const { setClosing } = useContext(AppStatusContext);

  useEffect(() => {
    const minimizeButton = document.getElementById('minimize');
    const restoreMaximizeButton = document.getElementById('restoreMaximize');
    const closeButton = document.getElementById('close');

    const handleResize = async () => {
      const maximized = await appWindow.isMaximized();
      setMaximized(maximized);
    };
    handleResize();

    window.addEventListener('resize', handleResize);

    const closeEmailWindow = async () => {
      setClosing(true);
    };

    const minimizeAction = () => {
      appWindow.minimize();
    };
    const restoreMaximizeAction = () => {
      appWindow.toggleMaximize();
    };
    const closeAction = async () => {
      await closeEmailWindow();
      const allWindows = await getAllWindows();
      allWindows.forEach(window => {
        window.close();
      });
    };
    if (minimizeButton) {
      minimizeButton.addEventListener('click', minimizeAction);
    }
    if (restoreMaximizeButton) {
      restoreMaximizeButton.addEventListener('click', restoreMaximizeAction);
    }
    if (closeButton) {
      closeButton.addEventListener('click', closeAction);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (minimizeButton) {
        minimizeButton.removeEventListener('click', minimizeAction);
      }
      if (restoreMaximizeButton) {
        restoreMaximizeButton.removeEventListener('click', restoreMaximizeAction);
      }
      if (closeButton) {
        closeButton.removeEventListener('click', closeAction);
      }
    };
  }, [setMaximized]);

  return (
    <div
      id="app-topbar"
      data-tauri-drag-region
      className="top-bar-container"
      data-testid="topbar-container"
      style={{
        '--topbar-container-background-color': assistant.header_bg_color,
      }}
    >
      <div
        className="logo-container"
        style={{
          '--logo-container-background-color': assistant.header_text_bg_color,
        }}
      >
        <div className="logo">
          <AssistantLogo />
        </div>
      </div>
      <div className="title-container" data-tauri-drag-region>
        <div
          className="aia-title"
          data-tauri-drag-region
          style={{ color: assistant.header_text_bg_color }}
        >
          {assistantName}
        </div>
      </div>
      <div>{children}</div>

      <div
        className="window-controls"
        style={{
          '--top_bar_container_bg_color': assistant.top_bar_container_bg_color,
        }}
      >
        <IconButton
          className="window-control"
          id="minimize"
          size="small"
          sx={{
            color: assistant.header_text_bg_color,
            '& .MuiSvgIcon-root': {
              color: assistant.header_text_bg_color,
            },
          }}
          data-testid="topbar-minimize-button"
        >
          <MinimizeIcon fontSize="small" />
        </IconButton>

        <IconButton
          className="window-control"
          id="restoreMaximize"
          size="small"
          sx={{
            color: assistant.header_text_bg_color,
            '& .MuiSvgIcon-root': {
              color: assistant.header_text_bg_color,
            },
          }}
          data-testid="topbar-maximize-button"
        >
          {isMaximized ? (
            <CloseFullscreenIcon fontSize="small" />
          ) : (
            <OpenInFullIcon fontSize="small" />
          )}
        </IconButton>

        <IconButton
          className="window-control"
          id="close"
          size="small"
          sx={{
            color: assistant.header_text_bg_color,
            '& .MuiSvgIcon-root': {
              color: assistant.header_text_bg_color,
            },
          }}
          data-testid="topbar-close-button"
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </div>
    </div>
  );
};

export default Topbar;
