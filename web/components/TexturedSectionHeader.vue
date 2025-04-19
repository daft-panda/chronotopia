<template>
    <div ref="containerRef" class="relative">
        <div v-if="loading" class="absolute inset-0 flex items-center justify-center" aria-live="polite">
            <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500" />
            <span class="sr-only">Loading texture...</span>
        </div>
        <!-- Accessible content - always present for screen readers -->
        <div class="sr-only" aria-live="polite">
            <h1>{{ title }}</h1>
            <p>{{ date }} at {{ time }}</p>
        </div>
        <!-- Fallback gradient rendering - visible when 3D fails -->
        <div v-if="error || fallbackMode" class="fallback-container absolute inset-0">
            <div :class="`gradient-bg ${textureType}`" class="absolute inset-0 w-full h-full" aria-hidden="true" />
            <div class="relative z-10 p-6 text-white">
                <h1 :class="`text-2xl font-bold fallback-text-${textureType}`">{{ title }}</h1>
                <div :class="`flex items-center mt-2 fallback-subtext-${textureType}`">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24"
                        stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>{{ date }} at {{ time }}</span>
                </div>
            </div>
        </div>
        <!-- 3D Canvas container - hidden from screen readers -->
        <div v-show="!error && !fallbackMode" ref="threeContainer" class="w-full h-full" aria-hidden="true" />
        <!-- Text overlay positioned above the 3D scene - hidden from screen readers -->
        <div v-show="!error && !fallbackMode" class="absolute top-0 left-0 w-full h-full pointer-events-none"
            aria-hidden="true">
            <div class="p-6 text-white text-shadow">
                <div class="opacity-0">{{ title }}</div>
                <div class="flex items-center mt-2 text-white text-shadow opacity-50">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24"
                        stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>{{ date }} at {{ time }}</span>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>
import type { PropType } from 'vue';
import { ref, onMounted, onBeforeUnmount, watch, computed } from 'vue';
import * as THREE from 'three';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

const props = defineProps({
    title: {
        type: String,
        required: true
    },
    date: {
        type: String,
        required: true
    },
    time: {
        type: String,
        required: true
    },
    textureType: {
        type: String as PropType<'asphalt' | 'forest-floor' | 'frozen-water-ice' | 'beach-sand'>,
        default: 'asphalt'
    },
    height: {
        type: Number,
        default: 128
    },
    /**
     * Force fallback mode for testing, performance reasons, or accessibility
     */
    forceFallback: {
        type: Boolean,
        default: false
    },
    /**
     * Use fallback mode for screen readers for better accessibility
     */
    useAccessibleFallback: {
        type: Boolean,
        default: true
    }
});

const containerRef = ref<HTMLElement | null>(null);
const threeContainer = ref<HTMLElement | null>(null);
const loading = ref(true);
const error = ref(false);
const fallbackMode = ref(false);
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let renderer: THREE.WebGLRenderer | null = null;
let textMesh: THREE.Mesh | null = null;

const texturePaths = computed(() => {
    const baseTexturePath = '/textures';

    const materialMaps = {
        asphalt: {
            baseColor: `${baseTexturePath}/asphalt/asphalt_0003_color_1k.jpg`,
            normal: `${baseTexturePath}/asphalt/asphalt_0003_normal_opengl_1k.png`,
            roughness: `${baseTexturePath}/asphalt/asphalt_0003_roughness_1k.jpg`,
            ao: `${baseTexturePath}/asphalt/asphalt_0003_ao_1k.jpg`,
            height: `${baseTexturePath}/asphalt/asphalt_0003_height_1k.png`,
            textDecal: `${baseTexturePath}/dirt-decal/decals_0008_color_1k.jpg`,
            textNormal: `${baseTexturePath}/dirt-decal/decals_0008_normal_opengl_1k.png`,
            textRoughness: `${baseTexturePath}/dirt-decal/decals_0008_roughness_1k.jpg`,
            textOpacity: `${baseTexturePath}/dirt-decal/decals_0008_opacity_1k.jpg`,
        },
        'forest-floor': {
            baseColor: `${baseTexturePath}/forest-floor/ground_0006_color_1k.jpg`,
            normal: `${baseTexturePath}/forest-floor/ground_0006_normal_opengl_1k.png`,
            roughness: `${baseTexturePath}/forest-floor/ground_0006_roughness_1k.jpg`,
            ao: `${baseTexturePath}/forest-floor/ground_0006_ao_1k.jpg`,
            height: `${baseTexturePath}/forest-floor/ground_0006_height_1k.png`,
            textDecal: `${baseTexturePath}/moss/ground_0014_color_1k.jpg`,
            textNormal: `${baseTexturePath}/moss/ground_0014_normal_opengl_1k.png`,
            textRoughness: `${baseTexturePath}/moss/ground_0014_roughness_1k.jpg`,
            textSubsurface: `${baseTexturePath}/moss/ground_0014_subsurface_1k.jpg`,
        },
        'frozen-water-ice': {
            baseColor: `${baseTexturePath}/frozen-water-ice/ground_0030_color_1k.jpg`,
            normal: `${baseTexturePath}/frozen-water-ice/ground_0030_normal_opengl_1k.png`,
            roughness: `${baseTexturePath}/frozen-water-ice/ground_0030_roughness_1k.jpg`,
            ao: `${baseTexturePath}/frozen-water-ice/ground_0030_ao_1k.jpg`,
            height: `${baseTexturePath}/frozen-water-ice/ground_0030_height_1k.png`,
            subsurface: `${baseTexturePath}/frozen-water-ice/ground_0030_subsurface_1k.jpg`,
            // Ice text will be procedurally generated
        },
        'beach-sand': {
            baseColor: `${baseTexturePath}/beach-sand/ground_0024_color_1k.jpg`,
            normal: `${baseTexturePath}/beach-sand/ground_0024_normal_opengl_1k.png`,
            roughness: `${baseTexturePath}/beach-sand/ground_0024_roughness_1k.jpg`,
            ao: `${baseTexturePath}/beach-sand/ground_0024_ao_1k.jpg`,
            height: `${baseTexturePath}/beach-sand/ground_0024_height_1k.png`,
            // Sand text will use the same material with adjustments
        }
    };

    return materialMaps[props.textureType];
});

// Check for WebGL support and accessibility preferences
const checkWebGLSupport = (): boolean => {
    // Check for reduced motion preference
    const prefersReducedMotion = typeof window !== 'undefined' &&
        window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // If user prefers reduced motion and we're set to use accessible fallback
    if ((prefersReducedMotion) && props.useAccessibleFallback) {
        return false;
    }

    // Otherwise check for WebGL support
    try {
        const canvas = document.createElement('canvas');
        return !!(
            window.WebGLRenderingContext &&
            (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
        );
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (e: unknown) {
        return false;
    }
};

// Initialize Three.js scene
const initThreeJS = async () => {
    if (!threeContainer.value) return;

    // Check if WebGL is supported
    if (!checkWebGLSupport() || props.forceFallback) {
        fallbackMode.value = true;
        loading.value = false;
        return;
    }

    try {
        // Create scene
        scene = new THREE.Scene();

        // Create camera
        const containerWidth = threeContainer.value.clientWidth;
        const containerHeight = threeContainer.value.clientHeight;
        camera = new THREE.PerspectiveCamera(
            35,
            containerWidth / containerHeight,
            0.1,
            1000
        );
        camera.position.set(0, 0, 2);
        camera.lookAt(0, 0, 0);

        // Create renderer
        renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        renderer.setSize(containerWidth, containerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        threeContainer.value.appendChild(renderer.domElement);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = true;
        scene.add(directionalLight);

        // Add a ground plane
        await createGroundPlane();

        // Add 3D text
        await create3DText();

        renderScene();

        // Mark loading as complete
        loading.value = false;
    } catch (e) {
        console.error('Error initializing Three.js:', e);
        error.value = true;
        fallbackMode.value = true;
        loading.value = false;
    }
};

// Create ground plane with PBR materials
const createGroundPlane = async () => {
    if (!scene) return;

    const textureLoader = new THREE.TextureLoader();
    const loadTexture = (path: string): Promise<THREE.Texture> => {
        return new Promise((resolve, reject) => {
            textureLoader.load(
                path,
                (texture) => {
                    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
                    texture.repeat.set(200, 200);
                    resolve(texture);
                },
                undefined,
                (err) => reject(err)
            );
        });
    };

    try {
        const paths = texturePaths.value;

        // Load all required textures
        const [
            baseColorMap,
            normalMap,
            roughnessMap,
            aoMap,
            heightMap,
        ] = await Promise.all([
            loadTexture(paths.baseColor),
            loadTexture(paths.normal),
            loadTexture(paths.roughness),
            loadTexture(paths.ao),
            loadTexture(paths.height),
        ]);

        // Create material
        const material = new THREE.MeshPhysicalMaterial({
            map: baseColorMap,
            normalMap: normalMap,
            roughnessMap: roughnessMap,
            aoMap: aoMap,
            displacementMap: heightMap,
            displacementScale: 0.05,
            side: THREE.DoubleSide,
        });

        // Create plane geometry
        const geometry = new THREE.PlaneGeometry(500, 500, 1, 1);
        geometry.setAttribute('uv2', geometry.attributes.uv); // For aoMap

        // Create mesh
        const plane = new THREE.Mesh(geometry, material);
        //plane.rotation.x = -Math.PI / 2;
        //plane.position.y = -0.2;
        plane.receiveShadow = true;
        scene.add(plane);

        // Add slight camera tilt for better perspective
        if (camera) {
            camera.position.y = 0.1;
            camera.position.z = 1.5;
            camera.lookAt(0, 0, 0);
        }
    } catch (e) {
        console.error('Error creating ground plane:', e);
        throw e;
    }
};

// Create 3D text with appropriate materials
const create3DText = async () => {
    if (!scene) return;

    try {
        // Load font
        const fontLoader = new FontLoader();
        const font = await fontLoader.loadAsync('/fonts/helvetiker_bold.typeface.json');

        // Create text geometry
        const geometry = new TextGeometry(props.title, {
            font: font,
            size: 0.2,
            depth: 0.01,
            curveSegments: 12,
            bevelEnabled: true,
            bevelThickness: 0.01,
            bevelSize: 0.005,
            bevelOffset: 0,
            bevelSegments: 4
        });

        geometry.computeBoundingBox();
        const textWidth = geometry.boundingBox ?
            geometry.boundingBox.max.x - geometry.boundingBox.min.x : 1;

        // Center text
        geometry.translate(-textWidth / 2, 0, 0);

        // Create text material based on texture type
        const material = await createTextMaterial();

        // Create mesh
        textMesh = new THREE.Mesh(geometry, material);
        textMesh.position.y = 0;
        textMesh.position.z = 0.01;
        textMesh.castShadow = true;
        textMesh.receiveShadow = true;
        scene.add(textMesh);
    } catch (e) {
        console.error('Error creating 3D text:', e);
        throw e;
    }
};

// Create appropriate text material based on texture type
const createTextMaterial = async () => {
    const textureLoader = new THREE.TextureLoader();
    const paths = texturePaths.value;

    switch (props.textureType) {
        case 'asphalt': {
            // Yellow road markings with dirt
            const [textDecal, textOpacity] = await Promise.all([
                textureLoader.loadAsync(paths.textDecal),
                textureLoader.loadAsync(paths.textOpacity),
            ]);

            return new THREE.MeshStandardMaterial({
                color: 0xf7d63e, // Yellow base
                map: textDecal,
                // alphaMap: textOpacity,
                transparent: true,
                roughness: 0.7,
                metalness: 0.1,
            });
        }

        case 'forest-floor': {
            // Moss text
            const [textDecal, textNormal, textRoughness] = await Promise.all([
                textureLoader.loadAsync(paths.textDecal),
                textureLoader.loadAsync(paths.textNormal),
                textureLoader.loadAsync(paths.textRoughness),
            ]);

            return new THREE.MeshStandardMaterial({
                map: textDecal,
                normalMap: textNormal,
                roughnessMap: textRoughness,
                roughness: 0.9,
                metalness: 0.0,
                color: 0x7ab35b, // Green tint
            });
        }

        case 'frozen-water-ice': {
            // Create procedural ice texture
            const iceTexture = createProceduralIceTexture();

            return new THREE.MeshPhysicalMaterial({
                color: 0xaadeff,
                roughness: 0.1,
                metalness: 0.0,
                transmission: 0.9, // Transparency
                thickness: 0.5,     // Refraction
                envMapIntensity: 1,
                clearcoat: 1.0,
                clearcoatRoughness: 0.1,
                ior: 1.4,
                specularIntensity: 1,
                specularColor: 0xffffff,
                envMap: iceTexture,
            });
        }

        case 'beach-sand': {
            // Use same sand texture but darker/less saturated
            const [baseColorMap, normalMap, roughnessMap] = await Promise.all([
                textureLoader.loadAsync(paths.baseColor),
                textureLoader.loadAsync(paths.normal),
                textureLoader.loadAsync(paths.roughness),
            ]);

            return new THREE.MeshStandardMaterial({
                map: baseColorMap,
                normalMap: normalMap,
                roughnessMap: roughnessMap,
                color: 0x443322, // Darker, desaturated version of sand
                roughness: 0.9,
                metalness: 0.0,
            });
        }

        default:
            return new THREE.MeshStandardMaterial({ color: 0xffffff });
    }
};

// Create procedural ice texture
const createProceduralIceTexture = (): THREE.Texture => {
    const size = 256;
    const data = new Uint8Array(size * size * 4);

    // Create noise pattern
    for (let i = 0; i < size * size * 4; i += 4) {
        // Base ice color (light blue)
        data[i] = 200 + Math.random() * 55; // R
        data[i + 1] = 220 + Math.random() * 35; // G
        data[i + 2] = 255; // B
        data[i + 3] = 255; // Alpha

        // Add some cracks
        const crackChance = Math.random();
        if (crackChance > 0.995) {
            data[i] = 255; // R
            data[i + 1] = 255; // G
            data[i + 2] = 255; // B
        }
    }

    // Create texture
    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(2, 2);
    texture.needsUpdate = true;

    return texture;
};

// Render scene once
const renderScene = () => {
    if (!scene || !camera || !renderer) return;

    // Set static position for text
    if (textMesh) {
        textMesh.rotation.y = 0;
        textMesh.position.y = 0.1;
    }

    // Render scene once
    renderer.render(scene, camera);
};

// Handle resize
const handleResize = () => {
    if (!renderer || !camera || !threeContainer.value) return;

    const width = threeContainer.value.clientWidth;
    const height = threeContainer.value.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
};

// Watch for texture type changes
watch(() => props.textureType, async () => {
    // Reinitialize scene with new textures
    if (scene) {
        // Reset scene
        if (textMesh) {
            scene.remove(textMesh);
            textMesh.geometry.dispose();
            if (Array.isArray(textMesh.material)) {
                textMesh.material.forEach(m => m.dispose());
            } else {
                textMesh.material.dispose();
            }
        }

        // Remove all other objects from scene
        while (scene.children.length > 0) {
            const object = scene.children[0];
            scene.remove(object);
            if (object instanceof THREE.Mesh) {
                object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(m => m.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            }
        }

        loading.value = true;
        try {
            // Add lights back
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            directionalLight.castShadow = true;
            scene.add(directionalLight);

            // Recreate ground and text
            await createGroundPlane();
            await create3DText();
            loading.value = false;
        } catch (e) {
            console.error('Error updating texture:', e);
            error.value = true;
            fallbackMode.value = true;
            loading.value = false;
        }
    }
});

// Watch for title changes
watch(() => props.title, async () => {
    if (scene && textMesh) {
        // Remove old text mesh
        scene.remove(textMesh);
        textMesh.geometry.dispose();
        if (Array.isArray(textMesh.material)) {
            textMesh.material.forEach(m => m.dispose());
        } else {
            textMesh.material.dispose();
        }

        // Create new text
        await create3DText();
    }
});

// Set container height
watch(() => props.height, () => {
    if (containerRef.value) {
        containerRef.value.style.height = `${props.height}px`;
    }
});

onMounted(async () => {
    if (containerRef.value) {
        containerRef.value.style.height = `${props.height}px`;
    }

    // Initialize Three.js
    await initThreeJS();

    // Add resize listener
    window.addEventListener('resize', handleResize);
});

onBeforeUnmount(() => {
    // Clean up Three.js resources
    if (renderer) {
        renderer.dispose();
    }

    if (scene) {
        scene.traverse((object) => {
            if (object instanceof THREE.Mesh) {
                object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(m => m.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            }
        });
    }

    // No need to cancel animation as we're not using an animation loop

    // Remove resize listener
    window.removeEventListener('resize', handleResize);
});
</script>

<style scoped>
.fallback-container {
    overflow: hidden;
}

.gradient-bg {
    background-size: 100% 100%;
}

.gradient-bg.asphalt {
    background: linear-gradient(to bottom, #333333, #222222);
}

.gradient-bg.forest-floor {
    background: linear-gradient(to bottom, #2c4a1b, #1a2e0e);
}

.gradient-bg.frozen-water-ice {
    background: linear-gradient(to bottom, #a8d8ff, #9bc5eb);
}

.gradient-bg.beach-sand {
    background: linear-gradient(to bottom, #e6d2a5, #d4b978);
}

.fallback-text-asphalt {
    color: #f7d63e;
    text-shadow: 0 0 5px rgba(255, 220, 0, 0.7);
}

.fallback-subtext-asphalt {
    color: #f0c000;
    text-shadow: 0 0 5px rgba(255, 220, 0, 0.5);
}

.fallback-text-forest-floor {
    color: #7ab35b;
    text-shadow: 0 0 8px rgba(100, 160, 70, 0.9);
}

.fallback-subtext-forest-floor {
    color: #8ed756;
    text-shadow: 0 0 5px rgba(100, 160, 70, 0.7);
}

.fallback-text-frozen-water-ice {
    color: #ffffff;
    text-shadow: 0 0 10px rgba(173, 216, 255, 0.9);
}

.fallback-subtext-frozen-water-ice {
    color: #ddefff;
    text-shadow: 0 0 8px rgba(173, 216, 255, 0.7);
}

.fallback-text-beach-sand {
    color: #443322;
    text-shadow: 0 1px 3px rgba(255, 255, 255, 0.5);
}

.fallback-subtext-beach-sand {
    color: #554433;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.4);
}

.text-shadow {
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
}
</style>