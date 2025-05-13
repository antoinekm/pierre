// apps/cli/src/commands/chat.ts
import { Command } from 'commander';
import chalk from 'chalk';
import inquirer from 'inquirer';
import { sendMessage } from '../client.js';
import { cancel, isCancel, log, outro, text } from '@clack/prompts';
import ora from 'ora';

export function registerChatCommand(program: Command): void {
  program
    .command('chat')
    .description('Start an interactive chat with Pierre')
    .action(async () => {
      log.info(chalk.blue(`Type "exit" to quit`));
      
      let chatActive = true;
      let conversationId = null;
      
      while (chatActive) {
        try {
          const message = await text({
            message: chalk.magenta('You:'),
            placeholder: 'Type your message here...',
          });

          if (isCancel(message)) {
            cancel('Chat session cancelled.');
            process.exit(0);
          }
          
          if (message.toString().toLowerCase() === 'exit') {
            outro(chalk.blue('Ending chat session.'));
            chatActive = false;
            process.exit(0);
          }
          
          const spinner = ora('Pierre is thinking...').start();
          
          // Set up event listeners for tool usage and progress updates
          const socket = global.socket;
          
          if (socket) {
            socket.once('tool-call', (data) => {
              spinner.text = `Using tool: ${data.toolName}`;
            });
            
            socket.once('tool-result', (data) => {
              spinner.text = 'Processing tool results...';
            });
            
            socket.once('text', (data) => {
              spinner.text = 'Generating response...';
            });
          }
          
          const response = await sendMessage(message.toString(), conversationId);
          conversationId = response.conversationId;
          
          spinner.stop();
          log.success(`${chalk.blue('Pierre:')}\n${response.text}`);
        } catch (error) {
          log.error(`Error in chat: ${error instanceof Error ? error.message : 'Unknown error'}`);
          
          const { continueChat } = await inquirer.prompt([
            {
              type: 'confirm',
              name: 'continueChat',
              message: 'Would you like to continue the chat?',
              default: true,
            },
          ]);
          
          if (!continueChat) {
            chatActive = false;
          }
        }
      }
    });
}