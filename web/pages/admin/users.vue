<template>
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-6">User Management</h1>

        <!-- Action buttons -->
        <div class="mb-6 flex justify-between items-center">
            <div class="flex gap-4 items-center">
                <button class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg shadow transition"
                    @click="openCreateUserModal">
                    Add New User
                </button>
                <div class="relative">
                    <input v-model="searchTerm" type="text" placeholder="Search users..."
                        class="pl-10 pr-4 py-2 border border-gray-300 rounded-lg w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        @input="debouncedSearch">
                    <span class="absolute left-3 top-2.5 text-gray-400">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24"
                            stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </span>
                </div>
            </div>
            <div class="flex items-center gap-2">
                <span class="text-gray-600">Sort by:</span>
                <select v-model="sortBy"
                    class="border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    @change="fetchUsers">
                    <option value="firstName">First Name</option>
                    <option value="lastName">Last Name</option>
                    <option value="email">Email</option>
                    <option value="dateCreated">Date Created</option>
                </select>
                <button
                    class="border border-gray-300 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    @click="toggleSortDirection">
                    <span v-if="sortDesc">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24"
                            stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
                        </svg>
                    </span>
                    <span v-else>
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24"
                            stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4" />
                        </svg>
                    </span>
                </button>
            </div>
        </div>

        <!-- Users Table -->
        <div class="bg-white rounded-lg shadow-md overflow-hidden mb-6">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                        </th>
                        <th scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Email
                        </th>
                        <th scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Username
                        </th>
                        <th scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Date Created
                        </th>
                        <th scope="col"
                            class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Actions
                        </th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    <tr v-if="isLoading" class="animate-pulse">
                        <td colspan="5" class="px-6 py-10 text-center text-gray-500">
                            Loading users...
                        </td>
                    </tr>
                    <tr v-else-if="users.length === 0" class="hover:bg-gray-50">
                        <td colspan="5" class="px-6 py-10 text-center text-gray-500">
                            No users found. Try adjusting your search criteria.
                        </td>
                    </tr>
                    <tr v-for="user in users" :key="user.id" class="hover:bg-gray-50">
                        <td class="px-6 py-4 whitespace-nowrap">
                            <div class="flex items-center">
                                <div
                                    class="flex-shrink-0 h-10 w-10 bg-gray-200 rounded-full flex items-center justify-center">
                                    <span class="text-gray-600 font-medium">{{ getUserInitials(user) }}</span>
                                </div>
                                <div class="ml-4">
                                    <div class="text-sm font-medium text-gray-900">{{ user.firstName }} {{ user.lastName
                                        }}</div>
                                </div>
                            </div>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ user.email }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ user.userName || '-' }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ formatDate(user.dateCreated) }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <div class="flex justify-end space-x-2">
                                <button class="text-blue-600 hover:text-blue-900" @click="editUser(user)">
                                    Edit
                                </button>
                                <button class="text-red-600 hover:text-red-900" @click="confirmDeleteUser(user)">
                                    Delete
                                </button>
                                <button class="text-green-600 hover:text-green-900" @click="resetPassword(user)">
                                    Reset Password
                                </button>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Pagination -->
        <div class="flex justify-between items-center">
            <div class="text-sm text-gray-700">
                Showing <span class="font-medium">{{ paginationInfo.from }}</span> to
                <span class="font-medium">{{ paginationInfo.to }}</span> of
                <span class="font-medium">{{ totalUsers }}</span> users
            </div>
            <div class="flex space-x-2">
                <button :disabled="currentPage === 1"
                    class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    @click="prevPage">
                    Previous
                </button>
                <button :disabled="currentPage === totalPages"
                    class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    @click="nextPage">
                    Next
                </button>
            </div>
        </div>

        <!-- Create User Modal -->
        <div v-if="showCreateUserModal"
            class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
                <h2 class="text-xl font-bold mb-4">{{ isEditMode ? 'Edit User' : 'Create New User' }}</h2>
                <form @submit.prevent="isEditMode ? updateUser() : createUser()">
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                            <input v-model="formData.firstName" type="text" required
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                            <input v-model="formData.lastName" type="text" required
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                            <input v-model="formData.email" type="email" required
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Username (Optional)</label>
                            <input v-model="formData.userName" type="text"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div v-if="!isEditMode">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Password</label>
                            <input v-model="formData.password" type="password" required minlength="8"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <p class="text-xs text-gray-500 mt-1">Password must be at least 8 characters and contain a
                                letter and a number.</p>
                        </div>
                    </div>
                    <div class="flex justify-end mt-6 space-x-2">
                        <button type="button"
                            class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                            @click="closeUserModal">
                            Cancel
                        </button>
                        <button type="submit" :disabled="isSubmitting"
                            class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md text-sm font-medium disabled:opacity-50">
                            {{ isSubmitting ? 'Saving...' : (isEditMode ? 'Update' : 'Create') }}
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <!-- Reset Password Modal -->
        <div v-if="showResetPasswordModal"
            class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
                <h2 class="text-xl font-bold mb-4">Reset Password for {{ selectedUser?.firstName }} {{
                    selectedUser?.lastName }}</h2>
                <form @submit.prevent="submitPasswordReset">
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                            <input v-model="passwordResetData.password" type="password" required minlength="8"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <p class="text-xs text-gray-500 mt-1">Password must be at least 8 characters and contain a
                                letter and a number.</p>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Confirm New Password</label>
                            <input v-model="passwordResetData.confirmPassword" type="password" required minlength="8"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                    </div>
                    <div class="flex justify-end mt-6 space-x-2">
                        <button type="button"
                            class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                            @click="closeResetPasswordModal">
                            Cancel
                        </button>
                        <button type="submit" :disabled="isSubmittingPassword || passwordsDoNotMatch"
                            class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md text-sm font-medium disabled:opacity-50">
                            {{ isSubmittingPassword ? 'Resetting...' : 'Reset Password' }}
                        </button>
                    </div>
                    <p v-if="passwordsDoNotMatch" class="text-red-500 text-sm mt-2">Passwords do not match.</p>
                </form>
            </div>
        </div>

        <!-- Delete Confirmation Modal -->
        <div v-if="showDeleteModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
                <h2 class="text-xl font-bold mb-4">Confirm Delete</h2>
                <p class="text-gray-600">
                    Are you sure you want to delete the user <strong>{{ selectedUser?.firstName }} {{
                        selectedUser?.lastName }}</strong>?
                    This action cannot be undone.
                </p>
                <div class="flex justify-end mt-6 space-x-2">
                    <button
                        class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                        @click="closeDeleteModal">
                        Cancel
                    </button>
                    <button :disabled="isDeleting"
                        class="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-md text-sm font-medium disabled:opacity-50"
                        @click="deleteUser">
                        {{ isDeleting ? 'Deleting...' : 'Delete User' }}
                    </button>
                </div>
            </div>
        </div>

        <!-- Toast Notification -->
        <div v-if="showToast"
            class="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg p-4 w-80 border-l-4 z-50 transition-opacity"
            :class="{
                'border-green-500': toastType === 'success',
                'border-red-500': toastType === 'error'
            }">
            <div class="flex">
                <div v-if="toastType === 'success'" class="flex-shrink-0 text-green-500">
                    <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                            clip-rule="evenodd" />
                    </svg>
                </div>
                <div v-else-if="toastType === 'error'" class="flex-shrink-0 text-red-500">
                    <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                            clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="ml-3">
                    <p class="text-sm font-medium text-gray-900">{{ toastMessage }}</p>
                </div>
                <div class="ml-auto pl-3">
                    <div class="-mx-1.5 -my-1.5">
                        <button class="inline-flex text-gray-400 hover:text-gray-500" @click="hideToast">
                            <span class="sr-only">Close</span>
                            <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"
                                fill="currentColor">
                                <path fill-rule="evenodd"
                                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                    clip-rule="evenodd" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>
import { ref, reactive, computed, onMounted, watch } from 'vue';
import { useDebounceFn } from '@vueuse/core';
import type { User } from '~/model/user_management_pb';
import type { Timestamp } from '@bufbuild/protobuf';

// Types
interface FormData {
    id?: string;
    firstName: string;
    lastName: string;
    email: string;
    userName?: string;
    password?: string;
}

interface PasswordResetData {
    password: string;
    confirmPassword: string;
}

// API client
const { userManagementApi } = useApi();
// State
const users = ref<User[]>([]);
const totalUsers = ref(0);
const totalPages = ref(1);
const currentPage = ref(1);
const pageSize = ref(10);
const isLoading = ref(false);
const searchTerm = ref('');
const sortBy = ref('lastName');
const sortDesc = ref(false);

// Form and modal state
const isEditMode = ref(false);
const showCreateUserModal = ref(false);
const showResetPasswordModal = ref(false);
const showDeleteModal = ref(false);
const isSubmitting = ref(false);
const isSubmittingPassword = ref(false);
const isDeleting = ref(false);
const selectedUser = ref<User | null>(null);
const formData = reactive<FormData>({
    firstName: '',
    lastName: '',
    email: '',
    userName: '',
    password: '',
});
const passwordResetData = reactive<PasswordResetData>({
    password: '',
    confirmPassword: '',
});

// Toast notifications
const showToast = ref(false);
const toastMessage = ref('');
const toastType = ref<'success' | 'error'>('success');
const toastTimeout = ref<number | null>(null);

// Computed properties
const paginationInfo = computed(() => {
    const from = (currentPage.value - 1) * pageSize.value + 1;
    const to = Math.min(from + pageSize.value - 1, totalUsers.value);

    return {
        from: totalUsers.value === 0 ? 0 : from,
        to,
    };
});

const passwordsDoNotMatch = computed(() => {
    return passwordResetData.password !== passwordResetData.confirmPassword;
});

// Methods
const fetchUsers = async () => {
    isLoading.value = true;

    try {
        const response = await userManagementApi.listUsers({
            searchTerm: searchTerm.value || undefined,
            page: currentPage.value,
            pageSize: pageSize.value,
            sortBy: sortBy.value,
            sortDesc: sortDesc.value,
        });

        users.value = response.users;
        totalUsers.value = response.totalCount;
        totalPages.value = response.totalPages;

        // Adjust current page if it's out of bounds after filter change
        if (currentPage.value > totalPages.value && totalPages.value > 0) {
            currentPage.value = 1;
            await fetchUsers();
        }
    } catch (error) {
        console.error('Failed to fetch users:', error);
        showErrorToast('Failed to load users. Please try again.');
    } finally {
        isLoading.value = false;
    }
};

const debouncedSearch = useDebounceFn(() => {
    currentPage.value = 1; // Reset to first page when searching
    fetchUsers();
}, 300);

const nextPage = () => {
    if (currentPage.value < totalPages.value) {
        currentPage.value++;
        fetchUsers();
    }
};

const prevPage = () => {
    if (currentPage.value > 1) {
        currentPage.value--;
        fetchUsers();
    }
};

const toggleSortDirection = () => {
    sortDesc.value = !sortDesc.value;
    fetchUsers();
};

const getUserInitials = (user: User) => {
    return `${user.firstName.charAt(0)}${user.lastName.charAt(0)}`;
};

const formatDate = (timestamp?: Timestamp) => {
    if (!timestamp) return 'N/A';

    const date = new Date(Number(timestamp.seconds) * 1000);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// Modal controls
const openCreateUserModal = () => {
    isEditMode.value = false;
    resetFormData();
    showCreateUserModal.value = true;
};

const editUser = (user: User) => {
    isEditMode.value = true;
    selectedUser.value = user;

    formData.id = user.id;
    formData.firstName = user.firstName;
    formData.lastName = user.lastName;
    formData.email = user.email;
    formData.userName = user.userName || '';
    delete formData.password; // No password field in edit mode

    showCreateUserModal.value = true;
};

const resetPassword = (user: User) => {
    selectedUser.value = user;
    passwordResetData.password = '';
    passwordResetData.confirmPassword = '';
    showResetPasswordModal.value = true;
};

const confirmDeleteUser = (user: User) => {
    selectedUser.value = user;
    showDeleteModal.value = true;
};

const closeUserModal = () => {
    showCreateUserModal.value = false;
    resetFormData();
};

const closeResetPasswordModal = () => {
    showResetPasswordModal.value = false;
    passwordResetData.password = '';
    passwordResetData.confirmPassword = '';
};

const closeDeleteModal = () => {
    showDeleteModal.value = false;
    selectedUser.value = null;
};

const resetFormData = () => {
    formData.id = undefined;
    formData.firstName = '';
    formData.lastName = '';
    formData.email = '';
    formData.userName = '';
    formData.password = '';
};

// CRUD operations
const createUser = async () => {
    isSubmitting.value = true;

    try {
        await userManagementApi.createUser({
            firstName: formData.firstName,
            lastName: formData.lastName,
            email: formData.email,
            userName: formData.userName || undefined,
            password: formData.password || '',
        });

        await fetchUsers();
        closeUserModal();
        showSuccessToast('User created successfully');
    } catch (error: any) {
        console.error('Failed to create user:', error);
        showErrorToast(getErrorMessage(error) || 'Failed to create user');
    } finally {
        isSubmitting.value = false;
    }
};

const updateUser = async () => {
    if (!formData.id) return;

    isSubmitting.value = true;

    try {
        const response = await userManagementApi.updateUser({
            id: formData.id,
            firstName: formData.firstName,
            lastName: formData.lastName,
            email: formData.email,
            userName: formData.userName || undefined,
        });

        // Update user in the list
        const index = users.value.findIndex(u => u.id === formData.id);
        if (index !== -1) {
            users.value[index] = response;
        }

        closeUserModal();
        showSuccessToast('User updated successfully');
    } catch (error: any) {
        console.error('Failed to update user:', error);
        showErrorToast(getErrorMessage(error) || 'Failed to update user');
    } finally {
        isSubmitting.value = false;
    }
};

const submitPasswordReset = async () => {
    if (!selectedUser.value) return;
    if (passwordsDoNotMatch.value) return;

    isSubmittingPassword.value = true;

    try {
        await userManagementApi.resetPassword({
            id: selectedUser.value.id,
            newPassword: passwordResetData.password,
        });

        closeResetPasswordModal();
        showSuccessToast('Password reset successful');
    } catch (error: any) {
        console.error('Failed to reset password:', error);
        showErrorToast(getErrorMessage(error) || 'Failed to reset password');
    } finally {
        isSubmittingPassword.value = false;
    }
};

const deleteUser = async () => {
    if (!selectedUser.value) return;

    isDeleting.value = true;

    try {
        await userManagementApi.deleteUser({
            id: selectedUser.value.id,
        });

        // Remove user from the list
        users.value = users.value.filter(u => u.id !== selectedUser.value?.id);
        totalUsers.value--;

        closeDeleteModal();
        showSuccessToast('User deleted successfully');

        // If the page is now empty and it's not the first page, go to previous page
        if (users.value.length === 0 && currentPage.value > 1) {
            currentPage.value--;
            await fetchUsers();
        }
    } catch (error: any) {
        console.error('Failed to delete user:', error);
        showErrorToast(getErrorMessage(error) || 'Failed to delete user');
    } finally {
        isDeleting.value = false;
    }
};

// Toast helpers
const showSuccessToast = (message: string) => {
    toastMessage.value = message;
    toastType.value = 'success';
    showToast.value = true;

    if (toastTimeout.value) {
        clearTimeout(toastTimeout.value);
    }

    toastTimeout.value = setTimeout(() => {
        hideToast();
    }, 5000);
};

const showErrorToast = (message: string) => {
    toastMessage.value = message;
    toastType.value = 'error';
    showToast.value = true;

    if (toastTimeout.value) {
        clearTimeout(toastTimeout.value);
    }

    toastTimeout.value = setTimeout(() => {
        hideToast();
    }, 5000);
};

const hideToast = () => {
    showToast.value = false;

    if (toastTimeout.value) {
        clearTimeout(toastTimeout.value);
        toastTimeout.value = null;
    }
};

// Helper to extract error messages from gRPC errors
const getErrorMessage = (error: any): string | null => {
    if (error.details) {
        return error.details;
    }
    if (error.message) {
        return error.message;
    }
    return null;
};

// Initialize
onMounted(() => {
    fetchUsers();
});

// Watch for changes in search term
watch(searchTerm, () => {
    debouncedSearch();
});

// Clean up
onUnmounted(() => {
    if (toastTimeout.value) {
        clearTimeout(toastTimeout.value);
    }
});

</script>