vim.pack.add({
  {
    src = 'https://github.com/nvim-neo-tree/neo-tree.nvim',
    version = vim.version.range('3')
  },
  "https://github.com/nvim-lua/plenary.nvim",
  "https://github.com/MunifTanjim/nui.nvim",
  "https://github.com/nvim-tree/nvim-web-devicons",
})

require("neo-tree").setup({
	close_if_last_window = true,
	follow_current_file = {
		enabled = true
	},
	filesystem = {
		filtered_items = {
			visible = true
		}
	}
})

vim.lsp.config['rust_ls'] = {
	cmd = { 'rust-analyzer' },
	filetypes = { 'rust' },
	root_markers = { '.git' }
}
vim.lsp.config['cpp_ls'] = {
	cmd = { 'clangd', '--compile-commands-dir=build' },
	filetypes = { 'cpp' },
	root_markers = { '.git' }
}
vim.lsp.enable('rust_ls')
vim.lsp.enable('cpp_ls')

vim.cmd(':colorscheme solarized')
vim.cmd(':set notermguicolors')
vim.cmd(':set number')
vim.cmd(':set tabstop=4')
vim.cmd(':set shiftwidth=4')
vim.cmd(':set expandtab')
vim.cmd(':set textwidth=120')
vim.cmd(':set colorcolumn=+1')

vim.keymap.set('n', '<leader>f', function()
    vim.lsp.buf.format()
end, {desc = 'formats the current buffer'})

vim.cmd(':below 15split term://bash')
vim.cmd(':tnoremap <Esc><Esc> <C-\\><C-n>')
vim.cmd(':tnoremap <A-h> <C-\\><C-N><C-w>h') 
vim.cmd(':tnoremap <A-j> <C-\\><C-N><C-w>j')
vim.cmd(':tnoremap <A-k> <C-\\><C-N><C-w>k')
vim.cmd(':tnoremap <A-l> <C-\\><C-N><C-w>l')
vim.cmd(':inoremap <A-h> <C-\\><C-N><C-w>h')
vim.cmd(':inoremap <A-j> <C-\\><C-N><C-w>j')
vim.cmd(':inoremap <A-k> <C-\\><C-N><C-w>k')
vim.cmd(':inoremap <A-l> <C-\\><C-N><C-w>l')
vim.cmd(':nnoremap <A-h> <C-w>h') 
vim.cmd(':nnoremap <A-j> <C-w>j')
vim.cmd(':nnoremap <A-k> <C-w>k')
vim.cmd(':nnoremap <A-l> <C-w>l')
