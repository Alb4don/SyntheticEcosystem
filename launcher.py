import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path.home() / '.digital_anthropology_lab'
PROJECT_MODULES = ('app', 'version', 'updates', 'logsafe', 'security', 'simulation', 'social', 'dialogue', 'observer', 'artificial_life', 'spatial', 'living_view', 'secure_store', 'local_store', 'learning', 'network_view', 'knowledge', 'knowledge_tool', 'layout', 'flow_layout', 'release_tool')


def _forget_modules():
    for name in PROJECT_MODULES:
        sys.modules.pop(name, None)


def choose_release():
    try:
        import logsafe
        import updates
        from version import __version__
        logsafe.configure(BASE)
        key = updates.load_trusted_key(HERE / 'update_key.pub')
        if key is None:
            return None
        return updates.UpdateManager(BASE, __version__, key).select_boot()
    except Exception:
        logging.getLogger('dal.launcher').exception('Falha ao selecionar versão; usando versão empacotada')
        return None


def main():
    sys.dont_write_bytecode = True
    release = choose_release()
    if release is not None:
        _forget_modules()
        sys.path.insert(0, str(release))
        try:
            import app
            return app.main(release_active=True, key_path=HERE / 'update_key.pub')
        except Exception:
            logging.getLogger('dal.launcher').exception('Versão atualizada falhou; revertendo para versão empacotada')
            sys.path.remove(str(release))
            _forget_modules()
    import app
    return app.main(release_active=False, key_path=HERE / 'update_key.pub')


if __name__ == '__main__':
    sys.exit(main())
