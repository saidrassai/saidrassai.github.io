# saidrassai.github.io

Personal space of Rassai Said.  
Inspired by: [nreHieW/rehiew.github.io](https://github.com/nreHieW/rehiew.github.io)

## Development
```bash
hugo server -D
hugo server --ignoreCache --disableFastRender --environment development
```

## Production
```bash
hugo -gc --minify --baseURL "https://nrehiew.github.io/blog/"
```

## CSS
```bash
cd themes/blog-theme
npm install
npm run watch
```