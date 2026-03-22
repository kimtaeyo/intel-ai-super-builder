import './Sidebar.css';
import React, { useState, useEffect, useContext, useRef, use } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { ChatContext } from '../context/ChatContext';
import { WorkflowContext } from '../context/WorkflowContext';
import useDataStore from '../../stores/DataStore';
import useMcpStore from '../../stores/McpStore';
import SettingsIcon from '@mui/icons-material/Settings';
import NewWorkflowIcon from '@mui/icons-material/AddCircle';
import HistoryIcon from '@mui/icons-material/History';
import AdminIcon from '@mui/icons-material/ManageAccounts';
import McpMarketplaceIcon from '@mui/icons-material/LocalMall';
import MCPIcon from '../../assets/mcpIcon/Mcpicon';
import { useTranslation } from 'react-i18next';
import McpManagement from '../mcpManagement/McpManagement';
import McpMarketPlace from '../mcpManagement/McpMarketPlace';
import { ChatHistory } from './ChatHistory';
import WorkflowOptions from '../workflows/WorkflowOptions';
import SidebarOverlay from './SidebarOverlay';
import Setting from '../setting/Setting';
import HelpIcon from '@mui/icons-material/Help';
import BugReportIcon from '@mui/icons-material/BugReport';
import HelpGuideIcon from '@mui/icons-material/School';
import LicenseInfoIcon from '@mui/icons-material/Policy';
import ModalWrapper from '../generalUseModal/generalUseModal';
import ModelLink from '../modelLink/ModelLink';
import { Typography, Link } from '@mui/material';
import { openUrl } from '@tauri-apps/plugin-opener';
import { exists } from '@tauri-apps/plugin-fs';
import { EmailWindowContext } from '../context/EmailWindowContext';

const Sidebar = ({}) => {
  const { t } = useTranslation();
  const { config, getDBConfig, system_info, assistant, assistantName } = useDataStore();
  const { newSession, isChatReady, newChatModelNeeded } = useContext(ChatContext);
  const { openBugReport } = useContext(EmailWindowContext);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const { workflowSidebarVisible: isWorkflowOpen, setWorkflowSidebarVisible: setIsWorkflowOpen } =
    useContext(WorkflowContext);
  const [isSettingOpen, setIsSettingOpen] = useState(false); // setting popout panel
  const [isHistoryOpen, setIsHistoryOpen] = useState(false); // chat history popout panel
  const sidebarRef = useRef(null);
  const [settingVisibility, setSettingVisibility] = useState(false);
  const mcpMarketplaceOpen = useMcpStore(state => state.mcpMarketplaceOpen);
  const mcpManagementOpen = useMcpStore(state => state.mcpManagementOpen);

  const handleSetAdmin = async () => {
    const shallowNewData = Object.keys(config).reduce((acc, key) => {
      if (typeof config[key] !== 'object' || config[key] === null) {
        acc[key] = config[key];
      }
      return acc;
    }, {});
    shallowNewData.is_admin = !config.is_admin;
    const viewModel = JSON.stringify(shallowNewData);
    await invoke('set_user_config_view_model', { vm: viewModel });
    getDBConfig();
  };

  const handleHistory = () => {
    setIsSettingOpen(false);
    setIsWorkflowOpen(false);
    setIsHistoryOpen(!isHistoryOpen);
    // Close marketplace when opening history
    useMcpStore.getState().closeMcpMarketplace();
    useMcpStore.getState().closeMcpManagement();
  };

  const handleSetting = () => {
    setIsHistoryOpen(false);
    setIsWorkflowOpen(false);
    setIsSettingOpen(!isSettingOpen);
    // Close marketplace when opening settings
    useMcpStore.getState().closeMcpMarketplace();
    useMcpStore.getState().closeMcpManagement();
  };

  const handleWorkflow = () => {
    setIsHistoryOpen(false);
    setIsSettingOpen(false);
    setIsWorkflowOpen(!isWorkflowOpen);
    // Close marketplace when opening workflow
    useMcpStore.getState().closeMcpMarketplace();
    useMcpStore.getState().closeMcpManagement();
  };

  const handleMarketplace = () => {
    setIsSettingOpen(false);
    setIsHistoryOpen(false);
    setIsWorkflowOpen(false);
    useMcpStore.getState().closeMcpManagement();
    if (!mcpMarketplaceOpen) {
      useMcpStore.getState().openMcpMarketplace();
    } else {
      useMcpStore.getState().closeMcpMarketplace();
    }
  };

  const handleManagement = () => {
    setIsSettingOpen(false);
    setIsHistoryOpen(false);
    setIsWorkflowOpen(false);
    useMcpStore.getState().closeMcpMarketplace();
    if (!mcpManagementOpen) {
      useMcpStore.getState().openMcpManagement();
    } else {
      useMcpStore.getState().closeMcpManagement();
    }
  };

  const closePanels = () => {
    setIsSettingOpen(false);
    setIsHistoryOpen(false);
    setIsWorkflowOpen(false);
    useMcpStore.getState().closeMcpMarketplace();
    useMcpStore.getState().closeMcpManagement();
  };

  const toggleInfo = () => setIsInfoOpen(!isInfoOpen);

  const openHelpGuide = async () => {
    await openUrl(
      'https://aibuilder.intel.com/Intel%20AI%20Assistant%20Builder%20User%20Guide.pdf'
    );
  };

  const openOSSFolder = async () => {
    try {
      let folderPath = 'C:\\Program Files\\Intel Corporation\\Intel(R) AI Super Builder\\oss_info';
      const pathExists = await exists(folderPath);
      if (!pathExists) {
        console.error('Path does not exist:', folderPath);
        return;
      }
      await invoke('open_in_explorer', { path: folderPath });
    } catch (error) {
      console.error('Error opening folder:', error);
    }
  };

  useEffect(() => {
    setSettingVisibility(config.is_admin);
  }, [config]);

  useEffect(() => {
    if (isChatReady == false && newChatModelNeeded == true && isSettingOpen == false) {
      handleSetting();
    }
  }, [isChatReady, newChatModelNeeded, isSettingOpen]);

  const isOpen =
    isSettingOpen || isHistoryOpen || isWorkflowOpen || mcpMarketplaceOpen || mcpManagementOpen;

  const SidebarBox = ({
    isChatReady,
    toggleSetting,
    toggleHistory,
    toggleWorkflow,
    toggleMarketplace,
    toggleManagement,
    toggleInfo,
    settingVisibility,
  }) => {
    const { t } = useTranslation();

    const SidebarButton = ({
      title,
      icon,
      onClick,
      additionalClasses = '',
      'data-testid': dataTestId,
    }) => {
      return (
        <button
          title={title}
          className={`sidebar-button ` + additionalClasses}
          onClick={onClick}
          disabled={!isChatReady}
          data-testid={dataTestId}
        >
          {icon}
        </button>
      );
    };

    return (
      <div className="sidebarbox" data-testid="sidebar-main-container">
        <SidebarButton
          additionalClasses="new-chat-button"
          title={t('sidebar.new_chat')}
          onClick={() => {
            if (isChatReady) {
              toggleWorkflow();
            }
          }}
          icon={
            <NewWorkflowIcon className="sidebar-icon" color="primary" sx={{ fontSize: '50px' }} />
          }
          data-testid="sidebar-new-chat-button"
        />
        <SidebarButton
          title={t('sidebar.chat_history')}
          onClick={() => {
            if (isChatReady) {
              toggleHistory();
            }
          }}
          icon={<HistoryIcon className="sidebar-icon" fontSize="large" />}
          data-testid="sidebar-chat-history-button"
        />
        {/* <SidebarButton
          title={t("sidebar.mcp_marketplace", "MCP Marketplace")}
          onClick={toggleMarketplace}
          icon={<McpMarketplaceIcon className="sidebar-icon" fontSize="large"/>}
          data-testid="sidebar-mcp-marketplace-button"
        />
        <SidebarButton
          title={t("sidebar.mcp_management", "MCP Management")}
          onClick={toggleManagement}
          icon={<MCPIcon className="sidebar-icon mcpmanagement-icon" fontSize="large" />}
          data-testid="sidebar-mcp-management-button"
        /> */}
        <div className="spacer"></div>
        <SidebarButton
          title={t('sidebar.help')}
          onClick={toggleInfo}
          icon={<HelpIcon className="sidebar-icon" fontSize="large" />}
          data-testid="sidebar-help-button"
        />
        {settingVisibility && (
          <SidebarButton
            title={t('sidebar.setting')}
            onClick={toggleSetting}
            icon={<SettingsIcon className="sidebar-icon" fontSize="large" />}
            data-testid="sidebar-settings-button"
          />
        )}
      </div>
    );
  };

  return (
    <>
      {isOpen && <div className="overlay" onClick={closePanels} />}
      <div className="sidebar-container" data-testid="sidebar-root" ref={sidebarRef}>
        <SidebarBox
          isChatReady={isChatReady}
          newSession={newSession}
          isSettingOpen={isSettingOpen}
          toggleSetting={() => handleSetting()}
          isHistoryOpen={isHistoryOpen}
          toggleHistory={() => handleHistory()}
          isWorkflowOpen={isWorkflowOpen}
          toggleWorkflow={() => handleWorkflow()}
          toggleMarketplace={() => handleMarketplace()}
          toggleManagement={() => handleManagement()}
          toggleInfo={toggleInfo}
          settingVisibility={settingVisibility}
        />
        <Setting
          isOpen={isSettingOpen}
          setIsOpen={setIsSettingOpen}
          onClose={() => setIsSettingOpen(false)}
        />
        <ModalWrapper
          isOpen={isInfoOpen}
          toggleOpen={toggleInfo}
          header={`${t('topbar.title')} - ${assistantName}`}
          hideFooter={false}
          buttonName={t('topbar.content_part_10')}
          data-testid="sidebar-info-modal"
        >
          <div>
            <div>
              <div>
                <Typography variant="body1">
                  {t('topbar.content_part_1')}{' '}
                  <Link href="https://www.intel.com/aipc" target="_blank" rel="noopener noreferrer">
                    {t('topbar.content_part_2')}
                  </Link>{' '}
                  {t('topbar.content_part_3')}
                </Typography>
                <br />
                <Typography variant="body1">
                  {t('topbar.content_part_4')}{' '}
                  <b>
                    v{system_info?.CurrentVersion} | {t('topbar.content_part_4_1')}{' '}
                    <Link
                      href="https://aibuilder.intel.com"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {t('topbar.content_part_5')}
                    </Link>
                  </b>
                </Typography>
                <Typography variant="body1">
                  {t('topbar.content_part_6')}{' '}
                  <b className="space-between">{assistant['models']['chat_model']}</b>
                  <b>
                    <ModelLink modelName={assistant['models']['chat_model']} />
                  </b>
                </Typography>
                <Typography variant="body1">
                  {t('topbar.content_part_7')}{' '}
                  <b className="space-between">{assistant['models']['embedding_model']}</b>
                  <b>
                    <ModelLink modelName={assistant['models']['embedding_model']} />
                  </b>
                </Typography>
                <Typography variant="body1">
                  {t('topbar.content_part_8')}{' '}
                  <b className="space-between">{assistant['models']['ranker_model']}</b>
                  <b>
                    <ModelLink modelName={assistant['models']['ranker_model']} />
                  </b>
                </Typography>
              </div>
              <br />
              <Link
                component="button"
                variant="body1"
                onClick={openBugReport}
                data-testid="sidebar-report-bug-button"
              >
                <BugReportIcon sx={{ pr: 1 }} />
                {'Report a bug'}
              </Link>
              <br />
              <Link
                component="button"
                variant="body1"
                onClick={openHelpGuide}
                data-testid="sidebar-help-guide-button"
              >
                <HelpGuideIcon sx={{ pr: 1 }} />
                {'View the user guide'}
              </Link>
              <br />
              <Link
                component="button"
                variant="body1"
                onClick={openOSSFolder}
                data-testid="sidebar-license-info-button"
              >
                <LicenseInfoIcon sx={{ pr: 1 }} />
                {t('topbar.content_part_9')}
              </Link>
            </div>
          </div>
        </ModalWrapper>
        <ChatHistory isOpen={isHistoryOpen} onClose={() => setIsHistoryOpen(false)} />
        <McpMarketPlace
          isOpen={mcpMarketplaceOpen}
          onClose={() => {
            if (isChatReady) {
              useMcpStore.getState().closeMcpMarketplace();
            }
          }}
        />
        {mcpManagementOpen && (
          <McpManagement
            isSidebarOpen={mcpManagementOpen}
            closePanels={() => {
              if (isChatReady) {
                useMcpStore.getState().closeMcpManagement();
              }
            }}
          />
        )}
        <SidebarOverlay
          isOpen={isWorkflowOpen}
          onClose={() => {
            if (isChatReady) {
              setIsWorkflowOpen(false);
            }
          }}
          content={
            <WorkflowOptions
              onWorkflowSelected={() => {
                setIsWorkflowOpen(false);
                // if (mcpManagementOpen) {
                //   useMcpStore.getState().closeMcpManagement();
                // }
              }}
            />
          }
        />
      </div>
    </>
  );
};

export default Sidebar;
