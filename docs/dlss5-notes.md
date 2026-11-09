# DLSS 5 en Hookshot: notas de trabajo

Estado a 2026-11-09: **aparcado**. Funciona y se ve bien, pero de momento no convence lo suficiente para seguir. Esta rama guarda
el trabajo para retomarlo. Base: original `2af3c2e` (v2.0.9-26), probado en Linux con RTX 5070 12 GB, driver 615.78.08, 2560x1440 a
180 Hz.

## Qué es

DLSS 5 no está en el SDK público de NVIDIA (`NVSDK_NGX_Feature_Reserved18`). Hookshot lo usa a través de
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (AGPL-3.0), una capa de Vulkan aparte con un proceso auxiliar bajo Wine o
Proton. Hookshot no enlaza la capa ni incluye nada de NVIDIA.

Cada jugador aporta `nvngx_dlssnr.dll` desde un juego suyo, porque no se puede redistribuir. La capa falsea la comprobación de NVIDIA
con `GetModuleFileNameW`; esa suplantación está en la capa y no debe escribirse dentro de Hookshot.

## Qué hay en esta rama

- Fila **DLSS 5** en Graphics (solo Linux). Activa la capa al arrancar.
- Con generación de fotogramas, la fila funciona **en caliente**: el hilo que procesa los fotogramas la lee en cada uno.
  Si el juego arrancó con la fila en Off, encenderla sigue pidiendo reinicio, porque la capa no se ha cargado.
- Fila **Smooth Motion**, generación de fotogramas del driver (RTX 40 y superiores).
- DLSS 5 **antes de la generación de fotogramas**: el modelo procesa solo los fotogramas reales y los intermedios salen de ya
  procesados. Usa `vkDlssnrProcessImageHOOKSHOT`, que **no existe en la capa publicada**.
- Segunda cola de presentación de la familia gráfica mientras DLSS 5 está activo. La cola propia del original
  (`RX_VK_PRESENT_QUEUE=1`) queda solo para cuando DLSS 5 está apagado.
- Cola corta: `kRingSlots = 3`, un solo fotograma esperando al modelo.
- **Modo demo**: `RX_VK_DLSS5_DEMO=<segundos>` alterna DLSS 5 y escribe `DLSS 5 ON/OFF` a la izquierda, a media altura, sin menú y
  sin contador de FPS.

## Dependencias sin publicar

Esta rama necesita la capa modificada. Sus cambios **no están en GitHub**: son las ramas locales `process-image` (entrada antes de
la generación de fotogramas, unas +213 líneas) y `real-motion` (además, profundidad y movimiento reales, unas +496). Habría que
proponérselas a su autor o publicar un fork AGPL.

Con la capa publicada, la función no se encuentra y DLSS 5 sigue en la presentación; el registro lo indica.

La rama local `feat/dlss-5-real-motion` (Hookshot +1.063 líneas, basada en `8b2e1d3`) añade profundidad y movimiento reales. Tampoco
se ha subido y no aportó más FPS.

## Mediciones

Salida 2560x1440, 3D y DLSS 5 al 100 %, pantalla completa. Unos minutos parado y andando en el bosque Kokiri, sobre `8b2e1d3`.

| Versión | Capa | FPS reales | FPS mostrados | p99 |
|---|---|---|---|---|
| Original, sin generación (*) | no | 60 (tope) | 60 | 19-20 ms |
| Original, x3 | no | 60 | 180 (tope) | 6,2 ms |
| 1. Smooth Motion | publicada | 41-43 | unos 80, sin medir | 25 ms |
| 2. Antes de x3 | `process-image` | 41-42 | 122-127 | 9 ms |
| 3. Movimiento real, x3 | `real-motion` | 41 | 122 | 9 ms |

(*) Las capturas de esa fila no valen: mostraban la terminal. Hay que repetirla.

- Las tres versiones con DLSS 5 cuestan lo mismo en FPS reales. Solo la 2 y la 3 llegan a 120 mostrados.
- Objetivo de 120 mostrados: conseguido con la versión 2. Objetivo de 60 reales: no conseguido.
- Modelo al 60 % de resolución: más FPS, pero borroso. Se mantiene al 100 %.
- Sin mejora: presets del modelo, resolución interna, pacing, memoria de texturas, pantalla completa, colas de presentación.
- MangoHud después de Smooth Motion en el orden de capas: `device lost` (Xid 32) en 2 de 2 pruebas.
- El modelo cuesta unos 13 ms a 1440p.

## Lo aprendido al usarlo

- **Intensidad**: se controla con el valor `intensity` de la capa. A 1 resulta demasiado realista para este juego; se probó 0,9,
  0,85, 0,8 y 0,5. **El 50 % es el que mejor se veía.** También existen `localtone` y `localstructure`, sin probar.
- **Retraso de entrada**: se nota al moverse. Estimado desde el código, sin medir, unos 100-120 ms antes de acortar la cola: hasta
  dos fotogramas esperando, el procesado, la interpolación que retiene un fotograma real y un grupo en espera de pantalla, a unos
  41 FPS reales.
- **Smooth Motion**: `vk_main.cpp` lo activa si la fila está en On aunque se use también generación de fotogramas, así que puede
  haber dos generadores encadenados. Sin confirmar en ejecución.
- La fila en caliente, la cola corta y el modo demo se probaron con el juego real, sin errores de GPU. No se midió el retraso.

## Ideas sin hacer

- Máscara del HUD para DLSS 5.
- Comprobar la imagen en movimiento y el parpadeo del mapa al andar.
- Medir de verdad el retraso, y qué aporta apagar Smooth Motion.
- Un lanzamiento en Windows: la capa no tiene versión para Windows.

## Cómo retomarlo

1. Actualizar esta rama sobre el original. Los conflictos suelen estar en `vk_framegen.cpp`, `vk_main.cpp` y `vk_replay.cpp`, en el
   código de presentación.
2. Compilar la capa con la rama `process-image` e instalarla.
3. Aportar `nvngx_dlssnr.dll` y arrancar el proceso auxiliar (`dlssnr-helper start`).
4. Poner **DLSS 5** en On, reiniciar, y fijar `intensity` en 0,5.
