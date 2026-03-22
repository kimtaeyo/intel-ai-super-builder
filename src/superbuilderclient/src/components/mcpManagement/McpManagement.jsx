import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Button,
  RadioGroup,
  FormControlLabel,
  Radio,
  FormControl,
  Typography,
  TextField,
  CircularProgress,
  Alert,
} from '@mui/material';
import ArrowCircleLeft from '@mui/icons-material/ArrowCircleLeft';
import TipsAndUpdatesOutlinedIcon from '@mui/icons-material/TipsAndUpdatesOutlined';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import StepContent from '@mui/material/StepContent';
import Autocomplete from '@mui/material/Autocomplete';
import './McpManagement.css';
import McpAgentTable from './McpAgentTable';
import McpServerTable from './McpServerTable';
import McpToolsDialog from './McpToolsDialog';
import FluidModal from '../FluidModal/FluidModal';
import useDataStore from '../../stores/DataStore';
import useMcpStore from '../../stores/McpStore';
import { useTranslation } from 'react-i18next';
import { ChatContext } from '../context/ChatContext';
import { McpTableHeader } from './mcpTableShared';
import McpServerInfo from './McpServerInfo';

/**
 * Validates environment variable format
 * Accepts: JSON format, KEY=VALUE pairs, KEY:VALUE pairs, or empty string
 */
const isValidEnvFormat = value => {
  const trimmed = value.trim();

  // Empty is valid (optional field)
  if (!trimmed) return true;

  // Check if it's valid JSON format
  if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
    try {
      const parsed = JSON.parse(trimmed);
      // Ensure it's an object (not array or other types)
      return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed);
    } catch {
      return false;
    }
  }

  // Check if it's KEY=VALUE or KEY:VALUE format (single or multiple pairs)
  // Pattern: KEY=VALUE or KEY:VALUE, optionally followed by spaces and more pairs
  const keyValuePattern = /^[A-Z_][A-Z0-9_]*[:=].+(\s+[A-Z_][A-Z0-9_]*[:=].+)*$/i;
  return keyValuePattern.test(trimmed);
};

const McpManagement = ({ isSidebarOpen = false, closePanels = () => {} }) => {
  const { t } = useTranslation();
  const { isChatReady, setIsChatReady } = useContext(ChatContext);
  const assistant = useDataStore(state => state.assistant);

  const mcpManagementOpen = useMcpStore(state => state.mcpManagementOpen);
  const mcpAgents = useMcpStore(state => state.mcpAgents);
  const mcpServers = useMcpStore(state => state.mcpServers);
  const selectedMcpServerNames = useMcpStore(state => state.selectedMcpServerNames);
  const selectedMcpServer = useMcpStore(state => state.selectedMcpServer);
  const selectedMcpAgent = useMcpStore(state => state.selectedMcpAgent);
  const mcpInputOpen = useMcpStore(state => state.mcpInputOpen);
  const mcpInputType = useMcpStore(state => state.mcpInputType);
  const mcpInputSource = useMcpStore(state => state.mcpInputSource);
  const mcpInput = useMcpStore(state => state.mcpInput);
  const mcpServerTools = useMcpStore(state => state.mcpServerTools);
  const runningMcpServers = useMcpStore(state => state.runningMcpServers);
  const loadingMcpServers = useMcpStore(state => state.loadingMcpServers);
  const fetchingMcpServerTools = useMcpStore(state => state.fetchingMcpServerTools);
  const mcpAgentInput = useMcpStore(state => state.mcpAgentInput);
  const mcpAgentInputOpen = useMcpStore(state => state.mcpAgentInputOpen);
  const mcpAgentInputType = useMcpStore(state => state.mcpAgentInputType);
  const mcpRemoveModalOpen = useMcpStore(state => state.mcpRemoveModalOpen);
  const mcpRemoveType = useMcpStore(state => state.mcpRemoveType);

  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [mcpInputError, setMcpServerInputError] = useState({});
  const [mcpAgentInputError, setMcpAgentInputError] = useState({});
  const [isEnvFieldFocused, setIsEnvFieldFocused] = useState(false); // Track env field focus state
  const envFieldRef = React.useRef(null); // Ref to env TextField
  const [verticalSplitPercent, setVerticalSplitPercent] = useState(null); // null = not yet calculated
  const [isResizing, setIsResizing] = useState(false);
  const [hasCalculatedInitialSplit, setHasCalculatedInitialSplit] = useState(false); // Track if we've done initial calculation with data
  const [isLoadingData, setIsLoadingData] = useState(false); // Track if initial data is being loaded
  const runningMcpAgents = useMcpStore(state => state.runningMcpAgents);
  const loadingMcpAgents = useMcpStore(state => state.loadingMcpAgents);
  const refreshTrigger = useMcpStore(state => state.refreshTrigger);

  const DEBUG_LAYOUT = false;

  // Calculate initial vertical split based on Agent table content
  // This runs whenever mcpAgents or mcpServers updates (after fetch completes)
  useEffect(() => {
    // Only calculate once, triggered by the first data update after opening the panel
    // The refreshTrigger ensures we recalculate if data is refreshed
    if (!hasCalculatedInitialSplit && mcpManagementOpen) {
      // DataGrid constants
      const HEADER_HEIGHT = 40; // Column header height
      const ROW_HEIGHT = 42; // Each row height
      const FOOTER_HEIGHT = 44; // Pagination footer
      const TABLE_HEADER_HEIGHT = 34; // McpTableHeader component height
      const RESIZE_HANDLE_HEIGHT = 6; // Resize handle height
      const GAP_HEIGHT = 12; // Gap between tables (combined with resize handle)
      const MARGIN_BOTTOM = 8; // Margin bottom of table container

      // App layout constants
      const APP_TITLE_BAR = 48; // App title bar height
      const PAGE_TOP_SECTION = 36; // Back button and toggle section (marginBottom: 8px)
      const PADDING = 20; // Container padding

      // Calculate Agent table minimum height
      // Using pageSize from pagination (default 10 rows)
      const defaultPageSize = 10;
      const numVisibleRows =
        mcpAgents.length === 0 ? 1 : Math.min(defaultPageSize, mcpAgents.length);
      const numVisibleServerRows =
        mcpServers.length === 0 ? 1 : Math.min(defaultPageSize, mcpServers.length);

      const agentTableMinHeight =
        TABLE_HEADER_HEIGHT +
        HEADER_HEIGHT +
        numVisibleRows * ROW_HEIGHT +
        FOOTER_HEIGHT +
        MARGIN_BOTTOM;

      // Calculate available container height from window height
      const windowHeight = window.innerHeight;
      const totalOverhead =
        APP_TITLE_BAR + PAGE_TOP_SECTION + PADDING + GAP_HEIGHT + RESIZE_HANDLE_HEIGHT;
      const availableHeight = windowHeight - totalOverhead;

      // Calculate percentage (agent table height / total available height)
      let calculatedPercent = (agentTableMinHeight / availableHeight) * 100;

      // Ensure both tables have enough space (min 200px each for DataGrid to function properly)
      const minTableHeight = 200;
      const minPercent = (minTableHeight / availableHeight) * 100;
      const maxPercent = 100 - minPercent;

      calculatedPercent = Math.max(minPercent, Math.min(maxPercent, calculatedPercent));

      const agentTableActualHeight = (availableHeight * calculatedPercent) / 100;
      const serverTableActualHeight = (availableHeight * (100 - calculatedPercent)) / 100;

      if (DEBUG_LAYOUT) {
        console.log('=== Vertical Layout Height Calculation ===');
        console.log('Window height:', windowHeight);
        console.log('App overhead:', totalOverhead);
        console.log('Available height:', availableHeight);
        console.log(
          'Agent data rows:',
          mcpAgents.length,
          '(allocating space for',
          numVisibleRows,
          'rows)'
        );
        console.log(
          'Server data rows:',
          mcpServers.length,
          '(allocating space for',
          numVisibleServerRows,
          'rows)'
        );
        console.log('Agent table min height needed:', agentTableMinHeight + 'px');
        console.log('Calculated percent:', calculatedPercent.toFixed(2) + '%');
        console.log('Agent table actual height:', agentTableActualHeight.toFixed(0) + 'px');
        console.log('Server table actual height:', serverTableActualHeight.toFixed(0) + 'px');
        console.log('Min table height enforced:', minTableHeight + 'px');
      }

      setVerticalSplitPercent(calculatedPercent);
      setHasCalculatedInitialSplit(true); // Mark that we've done the initial calculation
    }
  }, [mcpAgents, mcpServers, hasCalculatedInitialSplit, mcpManagementOpen, refreshTrigger]);

  // Helper function to create table container styles
  const createTableContainerStyle = verticalCalcOffset => ({
    height: `calc(100% - ${verticalCalcOffset}px)`,
    minHeight: '100px', // Minimum height for DataGrid to function
    overflow: 'auto',
    marginBottom: '0',
  });

  useEffect(() => {
    if (mcpManagementOpen) {
      const forceRefresh = refreshTrigger > 0;

      // Reset calculation flag when refreshing data
      if (forceRefresh) {
        setHasCalculatedInitialSplit(false);
        setVerticalSplitPercent(null);
      }

      // Check if we already have data in the store
      const hasData = mcpAgents.length > 0 || mcpServers.length > 0;

      setIsLoadingData(true);
      // Load data asynchronously
      // When getMcpAgent() and getLocalMcpServers() complete, they update the store
      // which will trigger the calculation useEffect below (since mcpAgents/mcpServers are dependencies)
      const loadData = async () => {
        try {
          if (!hasData || forceRefresh) {
            setIsLoadingData(true);
          }
          await Promise.all([
            useMcpStore.getState().getMcpAgent(forceRefresh),
            useMcpStore.getState().getActiveMcpAgents(),
            useMcpStore.getState().getLocalMcpServers(forceRefresh),
            useMcpStore.getState().getActiveMcpServers(),
          ]);
        } catch (err) {
          console.log('Error loading MCP management data:', err);
        } finally {
          setIsLoadingData(false);
        }
      };

      loadData();
    }
  }, [mcpManagementOpen, refreshTrigger]);

  useEffect(() => {
    if (mcpInputOpen) {
      setMcpServerInputError({
        mcpServerName: false,
        mcpServerNameDuplicate: false,
        mcpServerCommand: false,
        mcpServerCommandArgs: false,
        mcpServerUrl: false,
        mcpServerEnv: false,
      });
    }
  }, [mcpInputOpen]);

  useEffect(() => {
    if (mcpAgentInputOpen) {
      setMcpAgentInputError({
        agentName: false,
        agentNameDuplicate: false,
        description: false,
        systemMessage: false,
        mcpServerNames: false,
      });
    }
  }, [mcpAgentInputOpen]);

  // Resize handlers
  const handleMouseDown = e => {
    setIsResizing(true);
    e.preventDefault();
    e.stopPropagation();
  };

  React.useEffect(() => {
    const handleMouseMove = e => {
      const container = document.querySelector('.tables-container.vertical');
      if (!container) return;

      const containerRect = container.getBoundingClientRect();
      const containerHeight = containerRect.height;
      const mouseY = e.clientY - containerRect.top;
      const newHeightPercent = (mouseY / containerHeight) * 100;

      // Calculate min height in percentage (250px minimum for each panel, accounting for gap)
      const gap = 12; // gap between panels in pixels
      const minHeightPx = 250;
      const minHeightPercent = ((minHeightPx + gap) / containerHeight) * 100;
      const maxHeightPercent = 100 - minHeightPercent;

      // Constrain between min and max to ensure both panels stay at least 250px
      if (newHeightPercent >= minHeightPercent && newHeightPercent <= maxHeightPercent) {
        setVerticalSplitPercent(newHeightPercent);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.body.classList.add('resizing');
      document.addEventListener('mousemove', handleMouseMove, {
        passive: false,
      });
      document.addEventListener('mouseup', handleMouseUp, { passive: false });

      return () => {
        document.body.classList.remove('resizing');
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing]);

  const handleInputSourceChange = source => {
    useMcpStore.getState().setMcpInputSource(source);
    if (source === 'command') {
      useMcpStore.getState().setMcpInput({
        ...mcpInput,
        mcpServerUrl: '',
        mcpServerEnv: '',
      });
      setMcpServerInputError(prev => ({
        ...prev,
        mcpServerUrl: false,
      }));
    } else {
      useMcpStore.getState().setMcpInput({
        ...mcpInput,
        mcpServerCommand: '',
        mcpServerCommandArgs: '',
        mcpServerEnv: '',
      });
      setMcpServerInputError(prev => ({
        ...prev,
        mcpServerCommand: false,
        mcpServerCommandArgs: false,
      }));
    }
  };

  const handleManagementUIClose = () => {
    useMcpStore.getState().closeMcpManagement();
  };

  const handleInputModalClose = () => {
    setIsEnvFieldFocused(false); // Reset env field focus state when dialog closes
    useMcpStore.getState().closeMcpInput();
  };

  const handleInputModalOpen = type => {
    useMcpStore.getState().openMcpInput(type, 'command');
    if (type === 'Add') {
      useMcpStore.getState().setMcpInput({
        mcpServerName: '',
        mcpServerCommand: '',
        mcpServerCommandArgs: '',
        mcpServerUrl: '',
        mcpServerEnv: '',
        mcpServerDisabled: false,
      });
    }
  };

  const handleAgentInputModalClose = () => {
    useMcpStore.getState().closeMcpAgentInput();
  };

  const handleAgentInputModalOpen = type => {
    useMcpStore.getState().openMcpAgentInput(type);
    if (type === 'Add') {
      useMcpStore.getState().setMcpAgentInput({
        agentName: '',
        description: '',
        systemMessage: `You execute ONE assigned task in a workflow.

          INPUTS:
          - ORIGINAL QUESTION: Full user request (if provided - gives context)
          - DEPENDENCY OUTPUTS: Results from prerequisite tasks (if provided - your input data)
          - YOUR TASK: What you must do (ONLY this)

          INPUT PATTERNS:
          Pattern 1 (First step - no dependencies):
          - You receive: ORIGINAL QUESTION + YOUR TASK
          - Use ORIGINAL QUESTION to understand what data/action YOUR TASK needs

          Pattern 2 (Later step - has dependencies):
          - You receive: DEPENDENCY OUTPUTS + YOUR TASK
          - Use DEPENDENCY OUTPUTS as your input data
          - ORIGINAL QUESTION may not be provided (you don't need it)

          RULES:
          - Use tools as needed to complete YOUR TASK
          - If you have ORIGINAL QUESTION: understand context, but execute only YOUR TASK
          - If you have DEPENDENCY OUTPUTS: use them as input for YOUR TASK
          - Do not solve beyond YOUR TASK scope
          - Stop when YOUR TASK is done

          OUTPUT: When complete, simply report your results and say nothing else.`,
        mcpServerNames: [], // Initialize as empty array
      });
    } else if (type === 'Update') {
      // search for the selected agent and set editingAgentName
      const selected = useMcpStore.getState().selectedMcpAgent[0];
      if (selected) {
        useMcpStore.getState().setMcpAgentInput({
          ...selected,
          editingAgentName: selected.name,
          mcpServerNames: selected.server_names || [],
        });
      }
    }
  };

  const handleMcpServerSubmit = async type => {
    setIsChatReady(false);
    // Validate input before submission
    if (!Object.values(mcpInputError).every(v => v === false)) {
      return;
    }

    // Strip double quotes from mcpServerCommand
    if (mcpInputSource === 'command' && typeof mcpInput.mcpServerCommand === 'string') {
      const strippedCommand = mcpInput.mcpServerCommand.replace(/^"+|"+$/g, '');
      useMcpStore.getState().setMcpInput({
        ...mcpInput,
        mcpServerCommand: strippedCommand,
      });
    }

    let result;
    try {
      if (type === 'Add') {
        result = await useMcpStore.getState().addMcpServer();
      } else {
        result = await useMcpStore.getState().updateMcpServer();
      }

      if (result) {
        //wait for server list to refresh before closing the modal
        await useMcpStore.getState().getLocalMcpServers(true);
      }
    } catch (err) {
      console.error('Error submitting MCP server:', err);
    } finally {
      useMcpStore.getState().closeMcpInput();
      setIsChatReady(true);
    }
  };

  const namePattern = /^[A-Za-z0-9_-]+$/;

  const handleInputChange = field => event => {
    const value = field !== 'mcpServerDisabled' ? event.target.value : event.target.checked;

    if (field === 'mcpServerName') {
      setMcpServerInputError(prev => ({
        ...prev,
        mcpServerNameDuplicate:
          mcpInputType === 'Update'
            ? mcpServers.some(
                server => server.name === value.trim() && server.name !== mcpInput.editingServerName
              )
            : mcpServers.some(server => server.name === value.trim()),
        mcpServerName: !value.trim(),
        mcpServerNameInvalid: value.trim() && !namePattern.test(value.trim()),
      }));
    }

    if (mcpInputSource === 'url') {
      if (field === 'mcpServerUrl') {
        if (!value.trim()) {
          setMcpServerInputError(prev => ({
            ...prev,
            mcpServerUrl: true,
          }));
        } else {
          try {
            new URL(value);
            setMcpServerInputError(prev => ({
              ...prev,
              mcpServerUrl: false,
            }));
          } catch (e) {
            setMcpServerInputError(prev => ({
              ...prev,
              mcpServerUrl: true,
            }));
          }
        }
      }
    } else if (mcpInputSource === 'command') {
      if (field === 'mcpServerCommand') {
        setMcpServerInputError(prev => ({
          ...prev,
          [field]: !value.trim(),
        }));
      }
    }

    // Add validation for environment variables format
    if (field === 'mcpServerEnv') {
      // Validate environment variable format (JSON, KEY=VALUE, or KEY:VALUE)
      const isValid = isValidEnvFormat(value);
      setMcpServerInputError(prev => ({
        ...prev,
        mcpServerEnv: !isValid,
      }));
    }

    useMcpStore.getState().setMcpInput({
      ...mcpInput,
      [field]: value,
    });
  };

  const handleRemoveMcpServer = () => {
    setIsChatReady(false);
    useMcpStore.getState().removeMcpServer();
    setIsChatReady(true);
  };

  const handleEnvFieldFocus = () => {
    setIsEnvFieldFocused(true);

    // Auto-format JSON if the value is valid JSON
    // Use setTimeout to avoid interfering with focus event
    setTimeout(() => {
      const envValue = mcpInput.mcpServerEnv?.trim();
      if (envValue && envValue.startsWith('{') && envValue.endsWith('}')) {
        try {
          const parsed = JSON.parse(envValue);
          const formatted = JSON.stringify(parsed, null, 2);
          useMcpStore.getState().setMcpInput({
            ...mcpInput,
            mcpServerEnv: formatted,
          });
        } catch {
          // If not valid JSON, leave as is
        }
      }

      // Re-focus the field after state update to ensure blur works properly
      // This is needed regardless of JSON formatting
      setTimeout(() => {
        if (envFieldRef.current) {
          const input =
            envFieldRef.current.querySelector('textarea') ||
            envFieldRef.current.querySelector('input');
          if (input) {
            input.focus();
          }
        }
      }, 0);
    }, 0);
  };

  const handleEnvFieldBlur = () => {
    setIsEnvFieldFocused(false);
  };

  const handleMcpAgentSubmit = async type => {
    setIsChatReady(false);

    if (!Object.values(mcpAgentInputError).every(v => v === false)) {
      setIsChatReady(true);
      return;
    }

    try {
      let result;
      if (type === 'Add') {
        result = await useMcpStore.getState().addMcpAgent();
      } else {
        result = await useMcpStore.getState().updateMcpAgent();
      }

      if (result) {
        await useMcpStore.getState().getMcpAgent(true);
      }
    } catch (err) {
      console.error('Error saving MCP agent:', err);
    } finally {
      useMcpStore.getState().closeMcpAgentInput();
      setIsChatReady(true);
    }
  };

  const handleAgentInputChange = field => event => {
    const value = event.target.value;

    if (field === 'agentName') {
      setMcpAgentInputError(prev => ({
        ...prev,
        agentNameDuplicate:
          mcpAgentInputType === 'Update'
            ? mcpAgents.some(
                agent =>
                  agent.name === value.trim() && agent.name !== mcpAgentInput.editingAgentName
              )
            : mcpAgents.some(agent => agent.name === value.trim()),
        agentName: !value.trim(),
        agentNameInvalid: value.trim() && !namePattern.test(value.trim()),
      }));
    } else {
      setMcpAgentInputError(prev => ({
        ...prev,
        [field]: !value.trim(),
      }));
    }

    useMcpStore.getState().setMcpAgentInput({
      ...mcpAgentInput,
      [field]: value,
    });
  };

  const handleRemoveMcpAgent = async () => {
    setIsChatReady(false);
    await useMcpStore.getState().removeMcpAgent();
    setIsChatReady(true);
  };

  const handleOpenMarketplace = () => {
    useMcpStore.getState().closeMcpManagement();
    useMcpStore.getState().openMcpMarketplace();
  };

  // Render vertical layout (Agent table on top, Server table on bottom)
  const renderVerticalLayout = () => {
    // Use default 40% if calculation hasn't completed yet
    const topPanelHeight = verticalSplitPercent !== null ? verticalSplitPercent : 50;

    return (
      <Box
        className="tables-container vertical"
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          flex: 1,
          minHeight: 0,
          height: '100%',
          overflow: 'auto',
        }}
      >
        {/* Agent Table */}
        <Box
          className="filebox"
          sx={{
            display: 'flex',
            flexDirection: 'column',
            flex: '0 0 auto',
            height: `${topPanelHeight}%`,
            minHeight: '200px',
            overflow: 'auto',
          }}
        >
          <McpTableHeader
            title={'MCP Agents'}
            onAdd={() => handleAgentInputModalOpen('Add')}
            onRemove={() => {
              useMcpStore.getState().setMcpRemoveType('agent');
              useMcpStore.getState().setMcpRemoveModalOpen(true);
            }}
            addDisabled={!isChatReady}
            removeDisabled={
              !isChatReady ||
              selectedMcpAgent.length === 0 ||
              (selectedMcpAgent.length > 0 &&
                selectedMcpAgent.some(agent => runningMcpAgents.includes(agent.name)))
            }
            addButtonText={t('mcp.ui.add_agent_button')}
            removeButtonText={t('mcp.ui.header_remove_agent_button')}
            data-testid="mcp-agent"
          />
          <Box sx={createTableContainerStyle(34)}>
            <McpAgentTable />
          </Box>
        </Box>

        {/* Resize Handle - Vertical */}
        <Box
          className="resize-handle resize-handle-vertical"
          onMouseDown={handleMouseDown}
          sx={{
            height: '6px',
            minHeight: '6px',
            cursor: 'row-resize',
            backgroundColor: 'var(--divider-color)',
            position: 'relative',
            flexShrink: 0,
            zIndex: 1000,
            transition: 'background-color 0.15s ease',
            '&:hover': {
              backgroundColor: 'var(--primary-color)',
            },
            '&::after': {
              content: '""',
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '40px',
              height: '2px',
              backgroundColor: 'var(--text-secondary-color)',
              borderRadius: '1px',
              opacity: 0.5,
            },
          }}
        />

        {/* Server Table */}
        <Box
          className="filebox"
          sx={{
            display: 'flex',
            flexDirection: 'column',
            flex: '0 0 auto',
            height: `${100 - topPanelHeight}%`,
            minHeight: '200px',
            overflow: 'auto',
          }}
        >
          <McpTableHeader
            title={'MCP Servers'}
            onAdd={() => handleInputModalOpen('Add')}
            onRemove={() => {
              useMcpStore.getState().setMcpRemoveType('server');
              useMcpStore.getState().setMcpRemoveModalOpen(true);
            }}
            addDisabled={!isChatReady || loadingMcpServers.length > 0}
            removeDisabled={
              !isChatReady ||
              selectedMcpServer.length === 0 ||
              (selectedMcpServer.length > 0 &&
                (selectedMcpServer.some(server => runningMcpServers.includes(server.name)) ||
                  selectedMcpServerNames.some(id =>
                    mcpAgents.some(agent => agent.server_names && agent.server_names.includes(id))
                  )))
            }
            addButtonText={t('mcp.ui.add')}
            removeButtonText={t('mcp.ui.remove')}
            data-testid="mcp-server"
            enableAddMarketplace={true}
            isChatReady={isChatReady}
            handleOpenMarketplace={handleOpenMarketplace}
          />

          <Box sx={createTableContainerStyle(34)}>
            <McpServerTable />
          </Box>
        </Box>
      </Box>
    );
  };

  return (
    <>
      {isSidebarOpen && <div onClick={closePanels} />}
      <div
        className="mcp-modal-overlay"
        style={{ padding: '12px 24px' }}
        onClick={e => e.stopPropagation()}
      >
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            position: 'relative',
            gap: '8px',
          }}
        >
          <Button
            variant="contained"
            className="mcp-header-btn"
            onClick={handleManagementUIClose}
            data-testid="mcp-back-to-super-agent-btn"
          >
            <ArrowCircleLeft sx={{ fontSize: '18px' }} />
            {t('mcp.ui.back')}
          </Button>
          <Box
            sx={{
              position: 'absolute',
              left: 0,
              right: 0,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              pointerEvents: 'none',
            }}
          >
            <Typography
              variant="h5"
              component="h1"
              color="text.primary"
              sx={{ fontWeight: 'bold' }}
            >
              {t('sidebar.mcp_manager', 'MCP Manager')}
            </Typography>
            <Tooltip title={t('mcp.ui.guide_button', 'Quick Guide')}>
              <IconButton
                onClick={() => setIsGuideOpen(true)}
                size="small"
                data-testid="mcp-guide-btn"
                sx={{ color: '#f5a046', pointerEvents: 'auto', ml: 0.5 }}
              >
                <TipsAndUpdatesOutlinedIcon sx={{ fontSize: '22px' }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Tables Container */}
        <Box
          className={`${isResizing ? 'resizing' : ''}`}
          sx={{
            flex: 1,
            minHeight: 0,
            height: '100%',
            overflow: 'hidden',
            cursor: isResizing ? 'row-resize' : 'default',
            userSelect: isResizing ? 'none' : 'auto',
          }}
        >
          {isLoadingData ? (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                flexDirection: 'column',
                gap: 2,
              }}
            >
              <CircularProgress size={60} />
              <Typography variant="body1" color="text.secondary">
                {t('mcp.ui.loading', 'Loading MCP data...')}
              </Typography>
            </Box>
          ) : (
            renderVerticalLayout()
          )}
        </Box>
      </div>

      {/* Remove Modal */}
      <FluidModal
        open={mcpRemoveModalOpen}
        handleClose={() => useMcpStore.getState().setMcpRemoveModalOpen(false)}
        header={
          <strong style={{ color: 'var(--text-primary-color)' }}>
            {t('mcp.ui.confirm_remove')}{' '}
            {mcpRemoveType === 'server' ? t('mcp.ui.mcp_server') : t('mcp.ui.mcp_agent')}
          </strong>
        }
        width="40%"
        footer={
          <>
            <div className="mcpmodal-footer">
              <div className="button">
                <Button
                  size="m"
                  variant="text"
                  onClick={() => {
                    useMcpStore.getState().setMcpRemoveModalOpen(false);

                    if (mcpRemoveType === 'agent') {
                      useMcpStore.getState().setSelectedMcpAgent([]);
                    }
                  }}
                  data-testid={
                    mcpRemoveType === 'server'
                      ? 'mcp-modal-server-close-confirm-btn'
                      : 'mcp-modal-agent-close-confirm-btn'
                  }
                >
                  {t('mcp.ui.close_button')}
                </Button>
              </div>
              <div className="button">
                <Button
                  size="m"
                  variant="contained"
                  sx={{ backgroundColor: '#c73d3d' }}
                  onClick={() => {
                    if (mcpRemoveType === 'server') {
                      handleRemoveMcpServer();
                    } else {
                      handleRemoveMcpAgent();
                    }
                    useMcpStore.getState().setMcpRemoveModalOpen(false);
                  }}
                  data-testid={
                    mcpRemoveType === 'server'
                      ? 'mcp-modal-server-remove-confirm-btn'
                      : 'mcp-modal-agent-remove-confirm-btn'
                  }
                >
                  {t('mcp.ui.remove_button')}
                </Button>
              </div>
            </div>
          </>
        }
        assistant={assistant}
      >
        <div className="mcpmodal">
          <div className="mcpmodal-container">
            <div className="mcpmodal-content">
              <Typography component="div">
                {t('mcp.ui.confirm_remove_message')}{' '}
                {mcpRemoveType === 'server' ? t('mcp.ui.mcp_server') : t('mcp.ui.mcp_agent')}
                <br />
                {mcpRemoveType === 'server' ? (
                  <ul>
                    {useMcpStore.getState().selectedMcpServer.map(server => (
                      <li key={server.name}>{server.name}</li>
                    ))}
                  </ul>
                ) : (
                  <ul>
                    {useMcpStore.getState().selectedMcpAgent.map(agent => (
                      <li key={agent.name}>{agent.name}</li>
                    ))}
                  </ul>
                )}
              </Typography>
            </div>
          </div>
        </div>
      </FluidModal>

      {/* MCP Agent Modal */}
      <FluidModal
        open={mcpAgentInputOpen}
        handleClose={handleAgentInputModalClose}
        header={
          <strong>
            {mcpAgentInputType === 'Add' ? t('mcp.ui.add_agent') : t('mcp.ui.edit_agent')}
          </strong>
        }
        width="50%"
        footer={
          <>
            <div className="mcpmodal-footer">
              <div className="button">
                <Button
                  size="m"
                  variant="text"
                  onClick={handleAgentInputModalClose}
                  data-testid="mcp-agent-modal-close-btn"
                >
                  {t('mcp.ui.close_button')}
                </Button>
              </div>
              <div className="button">
                <Button
                  size="m"
                  variant="contained"
                  onClick={() => handleMcpAgentSubmit(mcpAgentInputType)}
                  disabled={
                    Object.values(mcpAgentInputError).some(error => error === true) ||
                    !(mcpAgentInput.agentName ?? '').trim() ||
                    !(mcpAgentInput.description ?? '').trim() ||
                    !(mcpAgentInput.systemMessage ?? '').trim() ||
                    !Array.isArray(mcpAgentInput.mcpServerNames) ||
                    runningMcpAgents.includes(mcpAgentInput.agentName)
                  }
                  data-testid={
                    mcpAgentInputType === 'Add'
                      ? 'mcp-agent-modal-add-btn'
                      : 'mcp-agent-modal-save-btn'
                  }
                >
                  {mcpAgentInputType === 'Add' ? t('mcp.ui.add_button') : t('mcp.ui.save_button')}
                </Button>
              </div>
            </div>
          </>
        }
        assistant={assistant}
      >
        <div className="mcpmodal">
          <div className="mcpmodal-container">
            <div className="mcpmodal-content">
              <Typography className="textfield-title">
                <span style={{ color: 'red' }}>*</span> {t('mcp.ui.mcp_agent_name')}
              </Typography>
              <TextField
                value={mcpAgentInput.agentName}
                disabled={
                  mcpAgentInputType === 'Update' &&
                  (loadingMcpAgents.includes(mcpAgentInput.agentName) ||
                    runningMcpAgents.includes(mcpAgentInput.agentName))
                }
                onChange={handleAgentInputChange('agentName')}
                fullWidth
                error={
                  mcpAgentInputError.agentName ||
                  mcpAgentInputError.agentNameDuplicate ||
                  mcpAgentInputError.agentNameInvalid
                }
                helperText={
                  mcpAgentInputError.agentName
                    ? 'MCP Agent Name is required'
                    : mcpAgentInputError.agentNameDuplicate
                      ? 'MCP Agent Name already exists'
                      : mcpAgentInputError.agentNameInvalid
                        ? 'Only letters, numbers, dashes, and underscores are allowed'
                        : ''
                }
                slotProps={{
                  formHelperText: {
                    sx: { color: 'red' },
                  },
                }}
                data-testid="mcp-agent-name-input"
              />
            </div>
            <div className="mcpmodal-content">
              <Typography className="textfield-title">
                <span style={{ color: 'red' }}>*</span> {t('mcp.ui.mcp_agent_description')}
              </Typography>
              <TextField
                value={mcpAgentInput.description}
                disabled={
                  mcpAgentInputType === 'Update' &&
                  (loadingMcpAgents.includes(mcpAgentInput.agentName) ||
                    runningMcpAgents.includes(mcpAgentInput.agentName))
                }
                onChange={handleAgentInputChange('description')}
                fullWidth
                error={mcpAgentInputError.description}
                helperText={
                  mcpAgentInputError.description ? 'MCP Agent Description is required' : ''
                }
                slotProps={{
                  formHelperText: {
                    sx: { color: 'red' },
                  },
                }}
                data-testid="mcp-agent-description-input"
              />
            </div>
            <div className="mcpmodal-content">
              <Typography className="textfield-title">
                <span style={{ color: 'red' }}>*</span> {t('mcp.ui.mcp_agent_system_prompt')}
              </Typography>
              <TextField
                value={mcpAgentInput.systemMessage}
                disabled={
                  mcpAgentInputType === 'Update' &&
                  (loadingMcpAgents.includes(mcpAgentInput.agentName) ||
                    runningMcpAgents.includes(mcpAgentInput.agentName))
                }
                onChange={handleAgentInputChange('systemMessage')}
                fullWidth
                error={mcpAgentInputError.systemMessage}
                helperText={
                  mcpAgentInputError.systemMessage ? 'MCP Agent System Prompt is required' : ''
                }
                slotProps={{
                  formHelperText: {
                    sx: { color: 'red' },
                  },
                }}
                data-testid="mcp-agent-system-message-input"
              />
            </div>
            <div className="mcpmodal-content">
              <Typography className="textfield-title">
                {t('mcp.ui.mcp_agent_mcp_server', 'MCP Servers')}
              </Typography>
              <FormControl fullWidth>
                <Autocomplete
                  multiple
                  options={mcpServers}
                  getOptionLabel={option => option.name}
                  value={
                    Array.isArray(mcpAgentInput.mcpServerNames) &&
                    mcpAgentInput.mcpServerNames.length > 0
                      ? mcpServers.filter(server =>
                          mcpAgentInput.mcpServerNames.includes(server.name)
                        )
                      : []
                  }
                  disabled={
                    mcpAgentInputType === 'Update' &&
                    (loadingMcpAgents.includes(mcpAgentInput.agentName) ||
                      runningMcpAgents.includes(mcpAgentInput.agentName))
                  }
                  onChange={(_, selected) => {
                    useMcpStore.getState().setMcpAgentInput({
                      ...mcpAgentInput,
                      mcpServerNames: selected.map(server => server.name), // Store as array of numbers
                    });
                  }}
                  renderOption={(props, option) => (
                    <li
                      {...props}
                      key={option.name}
                      data-testid={`mcp-server-option-${option.name}`} // add data-testid for per mcp server in table
                    >
                      {option.name}
                    </li>
                  )}
                  renderInput={params => (
                    <TextField
                      {...params}
                      variant="outlined"
                      placeholder={t('mcp.ui.mcp_agent_mcp_server_placeholder')}
                      helperText={
                        !mcpAgentInput.mcpServerNames || mcpAgentInput.mcpServerNames.length === 0
                          ? t('mcp.ui.mcp_agent_mcp_server_note')
                          : ''
                      }
                      data-testid="mcp-agent-mcp-servers-select"
                    />
                  )}
                />
              </FormControl>
            </div>
          </div>
        </div>
      </FluidModal>

      {/* MCP Server Modal */}
      <FluidModal
        open={mcpInputOpen}
        handleClose={handleInputModalClose}
        header={
          <strong style={{ color: 'var(--text-primary-color)' }}>
            {mcpInputType === 'Add' ? t('mcp.ui.add') : t('mcp.ui.edit')}
          </strong>
        }
        width={mcpInputType === 'Add' ? '40%' : '80%'}
        footer={
          <>
            <div className="mcpmodal-footer">
              <div className="button">
                <Button
                  size="m"
                  variant="text"
                  onClick={handleInputModalClose}
                  data-testid="mcp-server-modal-close-btn"
                >
                  {t('mcp.ui.close_button')}
                </Button>
              </div>
              <div className="button">
                <Button
                  size="m"
                  variant="contained"
                  onClick={() => handleMcpServerSubmit(mcpInputType)}
                  disabled={
                    Object.values(mcpInputError).some(error => error === true) ||
                    !(mcpInput.mcpServerName ?? '').trim() ||
                    (mcpInputSource === 'url' && !(mcpInput.mcpServerUrl ?? '').trim()) ||
                    (mcpInputSource === 'command' && !(mcpInput.mcpServerCommand ?? '').trim()) ||
                    loadingMcpServers.includes(mcpInput.mcpServerName) ||
                    runningMcpServers.includes(mcpInput.mcpServerName)
                  }
                  data-testid={
                    mcpInputType === 'Add'
                      ? 'mcp-server-modal-add-btn'
                      : 'mcp-server-modal-save-btn'
                  }
                >
                  {mcpInputType === 'Add' ? t('mcp.ui.add_button') : t('mcp.ui.save_button')}
                </Button>
              </div>
            </div>
          </>
        }
        assistant={assistant}
      >
        <div className="mcpmodal" style={{ maxHeight: '70vh' }}>
          <div className="mcpmodal-container">
            <FormControl
              component="fieldset"
              className="small-form-control"
              disabled={
                mcpInputType === 'Update' &&
                (loadingMcpServers.includes(mcpInput.mcpServerName) ||
                  runningMcpServers.includes(mcpInput.mcpServerName))
              }
            >
              {mcpInputType === 'Add' && (
                <>
                  {/* Docker MCP Server Recommendation Banner */}
                  <Alert
                    severity="info"
                    sx={{
                      width: '100%',
                      boxSizing: 'border-box',
                      mb: 2,
                      fontSize: '14px',
                    }}
                  >
                    {t('mcp.ui.docker_recommendation_description')}{' '}
                  </Alert>
                </>
              )}
              <div className="radio-group-with-label" sx={{ display: 'flex', width: '100%' }}>
                <RadioGroup
                  row
                  aria-label="model"
                  name="model"
                  value={mcpInputSource}
                  onChange={e => handleInputSourceChange(e.target.value)}
                >
                  <FormControlLabel
                    value="command"
                    data-testid="mcp-command-type-label"
                    control={<Radio color="primary" data-testid="mcp-command-type-radio" />}
                    label={t('mcp.ui.command_radio')}
                  />
                  <FormControlLabel
                    value="url"
                    data-testid="mcp-url-type-label"
                    control={<Radio color="primary" data-testid="mcp-url-type-radio" />}
                    label={t('mcp.ui.url_radio')}
                  />
                </RadioGroup>
              </div>
            </FormControl>

            <div className="mcpmodal-content">
              <Typography className="textfield-title" sx={{ color: 'var(--text-primary-color)' }}>
                <span style={{ color: 'red' }}>*</span> {t('mcp.ui.mcp_server_name')}
              </Typography>
              <TextField
                value={mcpInput.mcpServerName}
                placeholder=""
                onChange={handleInputChange('mcpServerName')}
                fullWidth
                disabled={
                  mcpInputType === 'Update' &&
                  (loadingMcpServers.includes(mcpInput.mcpServerName) ||
                    runningMcpServers.includes(mcpInput.mcpServerName))
                }
                error={
                  mcpInputError.mcpServerName ||
                  mcpInputError.mcpServerNameDuplicate ||
                  mcpInputError.mcpServerNameInvalid
                }
                helperText={
                  mcpInputError.mcpServerName
                    ? 'MCP Server Name is required'
                    : mcpInputError.mcpServerNameDuplicate
                      ? 'MCP Server Name already exists'
                      : mcpInputError.mcpServerNameInvalid
                        ? 'Only letters, numbers, dashes, and underscores are allowed'
                        : ''
                }
                slotProps={{
                  formHelperText: {
                    sx: { color: 'red' },
                  },
                }}
                data-testid="mcp-server-name-input"
              />
            </div>
            {mcpInputSource === 'command' ? (
              <>
                <div className="mcpmodal-content">
                  <Typography>
                    <span style={{ color: 'red' }}>*</span> {t('mcp.ui.mcp_server_command')}
                  </Typography>
                  <TextField
                    value={mcpInput.mcpServerCommand}
                    placeholder="e.g. docker"
                    onChange={handleInputChange('mcpServerCommand')}
                    fullWidth
                    disabled={
                      mcpInputType === 'Update' &&
                      (loadingMcpServers.includes(mcpInput.mcpServerName) ||
                        runningMcpServers.includes(mcpInput.mcpServerName))
                    }
                    error={mcpInputError.mcpServerCommand}
                    helperText={
                      mcpInputError.mcpServerCommand ? 'MCP Server Command is required' : ''
                    }
                    slotProps={{
                      formHelperText: {
                        sx: { color: 'red' },
                      },
                    }}
                    data-testid="mcp-server-command-input"
                  />
                </div>
                <div className="mcpmodal-content">
                  <Typography>{t('mcp.ui.mcp_server_command_args')}</Typography>
                  <TextField
                    value={mcpInput.mcpServerCommandArgs}
                    placeholder="e.g. run -i --rm mcp/time"
                    disabled={
                      mcpInputType === 'Update' &&
                      (loadingMcpServers.includes(mcpInput.mcpServerName) ||
                        runningMcpServers.includes(mcpInput.mcpServerName))
                    }
                    onChange={handleInputChange('mcpServerCommandArgs')}
                    fullWidth
                    data-testid="mcp-server-command-args-input"
                  />
                </div>
              </>
            ) : (
              <>
                <div className="mcpmodal-content">
                  <Typography>
                    <span style={{ color: 'red' }}>*</span> {t('mcp.ui.mcp_server_url')}
                  </Typography>
                  <TextField
                    value={mcpInput.mcpServerUrl}
                    placeholder="e.g. http://127.0.0.1:3008/sse"
                    onChange={handleInputChange('mcpServerUrl')}
                    fullWidth
                    disabled={
                      mcpInputType === 'Update' &&
                      (loadingMcpServers.includes(mcpInput.mcpServerName) ||
                        runningMcpServers.includes(mcpInput.mcpServerName))
                    }
                    error={mcpInputError.mcpServerUrl}
                    helperText={
                      mcpInputError.mcpServerUrl
                        ? !mcpInput.mcpServerUrl.trim()
                          ? 'MCP Server URL is required'
                          : 'MCP Server URL is invalid'
                        : ''
                    }
                    slotProps={{
                      formHelperText: {
                        sx: { color: 'red' },
                      },
                    }}
                    data-testid="mcp-server-url-input"
                  />
                </div>
              </>
            )}
            <div className="mcpmodal-content">
              <Typography>{t('mcp.ui.mcp_server_env')}</Typography>
              <TextField
                ref={envFieldRef}
                value={mcpInput.mcpServerEnv}
                placeholder={t('mcp.ui.mcp_server_env_placeholder')}
                onChange={handleInputChange('mcpServerEnv')}
                onFocus={handleEnvFieldFocus}
                onBlur={handleEnvFieldBlur}
                multiline={isEnvFieldFocused}
                rows={isEnvFieldFocused ? 6 : 1}
                fullWidth
                disabled={
                  mcpInputType === 'Update' &&
                  (loadingMcpServers.includes(mcpInput.mcpServerName) ||
                    runningMcpServers.includes(mcpInput.mcpServerName))
                }
                error={mcpInputError.mcpServerEnv}
                helperText={mcpInputError.mcpServerEnv ? t('mcp.ui.mcp_server_env_error') : ''}
                slotProps={{
                  formHelperText: {
                    sx: { color: 'red' },
                  },
                }}
                data-testid="mcp-server-env-input"
              />
            </div>
          </div>
          {mcpInputType === 'Update' && (
            <div className="mcpmodal-container">
              <Typography>{t('mcp.ui.mcp_server_info')}</Typography>
              <div className="mcpmodal-metadata">
                <McpServerInfo
                  serverName={mcpInput.mcpServerName}
                  serverConfig={mcpServers.find(s => s.name === mcpInput.mcpServerName)}
                  tools={mcpServerTools[mcpInput.mcpServerName] || []}
                  isLoading={fetchingMcpServerTools}
                  showServerName={false}
                  show="tools"
                />
              </div>
            </div>
          )}
        </div>
      </FluidModal>

      <McpToolsDialog />

      {/* Getting Started Guide Modal */}
      <FluidModal
        open={isGuideOpen}
        handleClose={() => setIsGuideOpen(false)}
        header={
          <strong style={{ color: 'var(--text-primary-color)' }}>
            {t('mcp.ui.guide_title', 'Getting Started with MCP Manager')}
          </strong>
        }
        width="45%"
        footer={
          <div className="mcpmodal-footer">
            <div className="button">
              <Button
                size="m"
                variant="contained"
                onClick={() => setIsGuideOpen(false)}
                data-testid="mcp-guide-close-btn"
              >
                {t('mcp.ui.close_button', 'Close')}
              </Button>
            </div>
          </div>
        }
        assistant={assistant}
      >
        <div className="mcpmodal" style={{ padding: '8px 0' }}>
          <Stepper orientation="vertical" nonLinear activeStep={-1}>
            <Step active>
              <StepLabel StepIconProps={{ sx: { fontSize: '28px', color: 'primary.main' } }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  {t('mcp.ui.guide_step1_title', 'Step 1 — Add MCP Server')}
                </Typography>
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary">
                  {t(
                    'mcp.ui.guide_step1_desc',
                    'Click the "Add MCP Server" button in the MCP Servers table. Fill in the server name and command. Make sure all required fields are correct before clicking "Add."'
                  )}
                </Typography>
              </StepContent>
            </Step>
            <Step active>
              <StepLabel StepIconProps={{ sx: { fontSize: '28px', color: 'primary.main' } }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  {t('mcp.ui.guide_step2_title', 'Step 2 — Add MCP Agent')}
                </Typography>
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary">
                  {t(
                    'mcp.ui.guide_step2_desc',
                    'Click the "Add Agent" button in the MCP Agents table. Give the agent a name and description, then select one or more MCP Servers from the list. At least one server must be selected to start the agent.'
                  )}
                </Typography>
              </StepContent>
            </Step>
            <Step active>
              <StepLabel StepIconProps={{ sx: { fontSize: '28px', color: 'primary.main' } }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  {t('mcp.ui.guide_step3_title', 'Step 3 — Start MCP Agent')}
                </Typography>
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary">
                  {t(
                    'mcp.ui.guide_step3_desc',
                    'Find your agent in the MCP Agents table and click the "Start" button. The agent will connect to its configured MCP servers and become ready for use in the chat.'
                  )}
                </Typography>
              </StepContent>
            </Step>
          </Stepper>
        </div>
      </FluidModal>
    </>
  );
};

export default McpManagement;
