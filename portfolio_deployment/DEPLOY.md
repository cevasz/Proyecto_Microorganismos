# Despliegue en Hugging Face Spaces

Este directorio es un **Static Space**. No requiere backend, GPU ni dependencias.
`index.html` es **autocontenido** (lleva los datos embebidos), por lo que basta con
subir ese archivo (más el `README.md` como tarjeta del Space).

## Archivos que se suben al Space

```
index.html   # visor completo con los escenarios embebidos
README.md     # tarjeta del Space (incluye metadata: sdk: static)
```

El resto de archivos (`viewer.html`, `scenario_*.json`, `cell_internal.json`,
`generate_data.py`, `generate_cell.py`, `build_index.py`, `DEPLOY.md`) son las
*fuentes* para regenerar el visor y **no** hacen falta en el Space.

## Opción A — Interfaz web (más simple)

1. Entra en <https://huggingface.co/new-space>.
2. **Space name**: `neurocolony-ec` · **SDK**: elige **Static** · **Visibility**: Public.
3. Crea el Space y, en la pestaña *Files*, pulsa **Add file → Upload files**.
4. Arrastra `index.html` y `README.md`.
5. *Commit*. El Space queda disponible en
   `https://huggingface.co/spaces/<tu-usuario>/neurocolony-ec`.

## Opción B — Git

```bash
# Requiere haber creado el Space (Static) en la web primero.
git clone https://huggingface.co/spaces/<tu-usuario>/neurocolony-ec
cd neurocolony-ec

cp /ruta/al/proyecto/portfolio_deployment/index.html .
cp /ruta/al/proyecto/portfolio_deployment/README.md .

git add .
git commit -m "NeuroColony-EC: visor interactivo"
git push
```

> `index.html` pesa ~5 MB (trayectorias de ambos escenarios + célula embebidas), muy
> por debajo del límite de subida. Si necesitas reducirlo, sube `save_every` (menos
> fotogramas) en `generate_data.py` y reconstruye.

## Regenerar los datos

Desde la raíz del proyecto (con el entorno virtual y los modelos entrenados):

```bash
python portfolio_deployment/generate_cell.py   # telemetría de la célula (LSTM real)
python portfolio_deployment/generate_data.py    # colonia (contexto de población) + index.html
```

Solo reconstruir `index.html` tras editar `viewer.html`:

```bash
cd portfolio_deployment && python build_index.py
```

## Probar el visor en local

`index.html` es autocontenido: **ábrelo con doble clic** (o arrástralo al navegador).
No necesita servidor.

> Nota: la plantilla `viewer.html` sí usa `fetch` y requiere un servidor
> (`python -m http.server`); es solo para desarrollo, no para desplegar.

Parámetros de URL soportados: `?view=colony|cell`, `?scenario=ppo|ext`, `?frame=N`, `?play=1`.
