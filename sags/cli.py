from nerfbaselines.cli import main
from nerfbaselines import MethodSpec, register

print("Registering sags train command...")

main.add_lazy_command('sags.train:train_command', 'train')
main.add_lazy_command('sags.render:render_command', 'render')

MethodSpec: MethodSpec = {
    "method_class": "sags.method:SAGS",
    "id": "sags",
}

register(MethodSpec)

print("Starting main CLI...")
main()