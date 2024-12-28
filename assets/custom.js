window.addEventListener('scroll', function () {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const buttons = document.querySelectorAll('#prev-button-preview, #next-button-preview');
    
    buttons.forEach(button => {
        button.style.top = '50%';
        button.style.transform = 'translateY(-50%)';
    });
});