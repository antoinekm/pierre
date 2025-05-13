// apps/cli/src/client.ts
import { io, Socket } from 'socket.io-client';
import { intro, log, outro } from '@clack/prompts';
import { Command } from 'commander';
import chalk from 'chalk';

export const DEFAULT_SERVER_URL = 'http://localhost:3333';
let socket: Socket | null = null;
let conversationId: string | null = null;
let hasInitialConnection = false;

declare global {
  var socket: Socket | null;
}

export async function initializeClient(program: Command): Promise<void> {
  if (process.argv.length <= 2 || 
      process.argv.includes('--help') || 
      process.argv.includes('-h') ||
      process.argv.includes('--version') ||
      process.argv.includes('-V')) {
    return Promise.resolve();
  }
  
  try {
    const serverUrl = process.env.PIERRE_SERVER_URL || DEFAULT_SERVER_URL;
    
    socket = io(serverUrl);
    global.socket = socket;

    const introMessage = chalk.bold(`⬢ ${program.name()} ${program.version()}`);
    
    socket.on('connect', () => {
      conversationId = `cli-session-${Date.now()}`;

      intro(introMessage);
      log.message(`- Server:  ${serverUrl}
- Session: ${conversationId}`);
      
      hasInitialConnection = true;
    });
    
    socket.on('connect_error', (error) => {
      if (!hasInitialConnection) {
        intro(introMessage);
      }
      
      log.error(`Failed to connect to server: ${error.message}`);
      outro('Please check if the server is running and try again.');
      process.exit(1);
    });
    
    socket.on('text', (data) => {
      // This event is just for progress updates, no need to display
    });
    
    socket.on('tool-call', (data) => {
      log.info(chalk.yellow(`Tool called: ${data.toolName}`));
    });
    
    socket.on('tool-result', (data) => {
      log.info(chalk.green(`Tool execution completed`));
    });
    
    socket.on('disconnect', (reason) => {
      if (reason === 'io server disconnect' || reason === 'io client disconnect') {
        outro('See you next time!');
      } else {
        log.error(`Connection lost: ${reason}`);
        outro('Please check if the server is still running and try again.');
        process.exit(1);
      }
      
      conversationId = null;
    });
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
      }, 5000);
      
      socket!.once('connect', () => {
        clearTimeout(timeout);
        resolve();
      });
      
      socket!.once('connect_error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
  } catch (error) {
    log.error(`Failed to initialize client: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
}

export async function closeClient(): Promise<void> {
  if (socket) {
    socket.disconnect();
    socket = null;
    global.socket = null;
    conversationId = null;
  }
}

export async function sendMessage(message: string, convId: string | null = null): Promise<{text: string, conversationId: string}> {
  return new Promise((resolve, reject) => {
    if (!socket || !socket.connected) {
      reject(new Error('Not connected to Pierre server'));
      return;
    }
    
    if (!conversationId && !convId) {
      conversationId = `cli-session-${Date.now()}`;
      log.info(`Created new conversation with ID: ${conversationId}`);
    } else if (convId) {
      conversationId = convId;
    }
    
    socket.emit('message', { 
      message,
      conversationId,
      timestamp: Date.now()
    });
    
    socket.once('response', (data) => {
      resolve({
        text: data.response,
        conversationId: data.conversationId
      });
    });
    
    socket.once('error', (data) => {
      reject(new Error(data.error || 'Unknown error'));
    });
    
    setTimeout(() => {
      reject(new Error('Response timeout'));
    }, 120000); // 2 minute timeout for multi-step processes
  });
}