// apps/core/src/remote/server.ts
import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import path from 'path';
import { logger } from '../utils/logger.js';
import { getAssistantResponse } from '../ai/llm.js';
import { authenticate } from '../security/auth.js';
import { 
  initializeConversationManager,
  processConversationMessage,
  addConversationResource
} from '../ai/conversation.js';
import { executeAgentLoop, streamAgentLoop } from '../ai/agentLoop.js';

// Define proper types for the server and io instances
let io: Server;
let server: http.Server;

export async function startServer(port: number, host: string): Promise<void> {
  try {
    const app = express();
    server = http.createServer(app);
    
    initializeConversationManager();
    
    io = new Server(server, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST'],
      },
    });
    
    app.use('/api', authenticate);
    app.use(express.static(path.join(process.cwd(), 'public')));
    
    app.post('/api/message', express.json(), async (req, res) => {
      try {
        const { message, conversationId } = req.body;
        
        if (!message) {
          return res.status(400).json({ error: 'Message is required' });
        }
        
        const actualConversationId = conversationId || `api-${Date.now()}`;
        
        let response;
        if (conversationId) {
          response = await processConversationMessage(actualConversationId, message);
        } else {
          response = await getAssistantResponse(message);
        }
        
        return res.json({ response, conversationId: actualConversationId });
      } catch (error) {
        logger.error('Error processing message:', error);
        return res.status(500).json({ error: 'Failed to process message' });
      }
    });
    
    app.post('/api/stream', express.json(), async (req, res) => {
      try {
        const { message, conversationId } = req.body;
        
        if (!message) {
          return res.status(400).json({ error: 'Message is required' });
        }
        
        const actualConversationId = conversationId || `api-stream-${Date.now()}`;
        
        // Set up streaming headers
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        // Use the streaming agent loop with callbacks
        await streamAgentLoop(message, 10, {
          onText: (text) => {
            res.write(`data: ${JSON.stringify({ type: 'text', content: text, conversationId: actualConversationId })}\n\n`);
          },
          onToolCall: (toolName, args) => {
            res.write(`data: ${JSON.stringify({ type: 'tool-call', toolName, args, conversationId: actualConversationId })}\n\n`);
            
            if (conversationId) {
              addConversationResource(actualConversationId, 'toolResult', {
                toolName, 
                args,
                result: null // We don't have the result yet
              });
            }
          },
          onToolResult: (result) => {
            res.write(`data: ${JSON.stringify({ type: 'tool-result', result, conversationId: actualConversationId })}\n\n`);
            
            if (conversationId) {
              addConversationResource(actualConversationId, 'toolResult', {
                toolName: result.toolName,
                args: result.args,
                result: result.result
              });
            }
          },
          onComplete: (finalText) => {
            res.write(`event: done\ndata: ${JSON.stringify({ text: finalText, conversationId: actualConversationId })}\n\n`);
            res.end();
          }
        }).catch(error => {
          logger.error('Error in streaming agent loop:', error);
          res.write(`event: error\ndata: ${JSON.stringify({ error: 'Stream processing error' })}\n\n`);
          res.end();
        });
      } catch (error) {
        logger.error('Error initializing stream:', error);
        return res.status(500).json({ error: 'Failed to initialize stream' });
      }
    });
    
    io.on('connection', (socket) => {
      logger.info(`New client connected: ${socket.id}`);
      
      socket.on('message', async (data) => {
        try {
          const { message, conversationId } = data;
          const actualConversationId = conversationId || `socket-${socket.id}-${Date.now()}`;
          
          logger.info(`Message from client ${socket.id} (conversation: ${actualConversationId}): ${message}`);
          
          // Set up streaming with Socket.IO
          try {
            await streamAgentLoop(message, 10, {
              onText: (text) => {
                socket.emit('text', { text, conversationId: actualConversationId });
              },
              onToolCall: (toolName, args) => {
                socket.emit('tool-call', { toolName, args, conversationId: actualConversationId });
                
                if (conversationId) {
                  addConversationResource(actualConversationId, 'toolResult', {
                    toolName, 
                    args,
                    result: null // We don't have the result yet
                  });
                }
              },
              onToolResult: (result) => {
                socket.emit('tool-result', { result, conversationId: actualConversationId });
                
                if (conversationId) {
                  addConversationResource(actualConversationId, 'toolResult', {
                    toolName: result.toolName,
                    args: result.args,
                    result: result.result
                  });
                }
              },
              onComplete: (finalText) => {
                socket.emit('response', { 
                  response: finalText,
                  conversationId: actualConversationId,
                  timestamp: Date.now()
                });
              }
            });
          } catch (error) {
            logger.error('Error in streaming agent loop:', error);
            socket.emit('error', { error: 'Stream processing error' });
          }
          
        } catch (error) {
          logger.error('Error handling socket message:', error);
          socket.emit('error', { error: 'Failed to process message' });
        }
      });
      
      socket.on('screenshot_captured', (data) => {
        const { conversationId, screenshotPath, description } = data;
        
        if (conversationId && screenshotPath) {
          addConversationResource(conversationId, 'screenshot', {
            path: screenshotPath,
            description: description || 'Screenshot captured by user'
          });
        }
      });
      
      socket.on('disconnect', () => {
        logger.info(`Client disconnected: ${socket.id}`);
      });
    });
    
    server.listen(port, host, () => {
      logger.info(`Remote server started at http://${host}:${port}`);
    });
  } catch (error) {
    logger.error('Error starting remote server:', error);
    throw error;
  }
}

export async function stopServer(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!server) {
      resolve();
      return;
    }
    
    server.close((err: Error | undefined) => {
      if (err) {
        logger.error('Error stopping server:', err);
        reject(err);
      } else {
        logger.info('Remote server stopped');
        resolve();
      }
    });
  });
}