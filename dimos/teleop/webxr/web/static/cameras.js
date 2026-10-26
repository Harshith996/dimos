const status = document.getElementById('status');
const container = document.getElementById('cameras');
const labels = { egocentric: 'Front', left_wrist: 'Left wrist', right_wrist: 'Right wrist' };
let panels = [];

async function start() {
    try {
        const response = await fetch('/teleop/config');
        if (!response.ok) throw new Error(`Camera config: HTTP ${response.status}`);
        const config = await response.json();
        panels = config.camera_panels.map(({ name }) => {
            const figure = document.createElement('figure');
            const caption = document.createElement('figcaption');
            const image = document.createElement('img');
            caption.textContent = labels[name] ?? name;
            image.alt = `${caption.textContent} camera`;
            figure.append(caption, image);
            container.append(figure);
            return image;
        });
        if (panels.length === 0) {
            status.textContent = 'No cameras are configured for this session.';
            return;
        }
        connect();
    } catch (error) {
        status.textContent = `Unable to load cameras: ${error.message}`;
    }
}

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/cameras/ws`);
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => { status.textContent = 'Live'; };
    socket.onmessage = ({ data }) => {
        if (!(data instanceof ArrayBuffer) || data.byteLength < 2) return;
        const bytes = new Uint8Array(data);
        const image = panels[bytes[0]];
        if (!image) return;
        const nextUrl = URL.createObjectURL(new Blob([bytes.subarray(1)], { type: 'image/jpeg' }));
        const previousUrl = image.dataset.url;
        image.src = nextUrl;
        image.dataset.url = nextUrl;
        if (previousUrl) URL.revokeObjectURL(previousUrl);
    };
    socket.onclose = () => {
        status.textContent = 'Disconnected. Reconnecting…';
        setTimeout(connect, 2000);
    };
    socket.onerror = () => { socket.close(); };
}

window.addEventListener('pagehide', () => {
    for (const image of panels) if (image.dataset.url) URL.revokeObjectURL(image.dataset.url);
});
start();
