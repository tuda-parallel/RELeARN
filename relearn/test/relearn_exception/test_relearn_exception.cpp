/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_relearn_exception.h"

#include "util/RelearnException.h"

#include <string>

TEST_F(RelearnExceptionTest, testFail) {
    RelearnException::hide_messages = false;

    const std::string message = "sadflhbcn\nkow97430921*:)�\" $SDMFSL ";
    ASSERT_THROW(RelearnException::fail(message), RelearnException);

    try {
        RelearnException::fail(message);
    } catch (const RelearnException& ex) {
        const auto& reason = ex.what();
        ASSERT_EQ(message, reason);
    }

    RelearnException::hide_messages = true;
}

TEST_F(RelearnExceptionTest, testCheck) {
    RelearnException::hide_messages = false;

    const std::string message = "sadflhbcn\nkow97430921*:)�\" $SDMFSL ";

    ASSERT_NO_THROW(RelearnException::check(true, message));
    ASSERT_THROW(RelearnException::check(false, message), RelearnException);

    try {
        RelearnException::check(false, message);
    } catch (const RelearnException& ex) {
        const auto& reason = ex.what();
        ASSERT_EQ(message, reason);
    }

    RelearnException::hide_messages = true;
}

TEST_F(RelearnExceptionTest, testFormatting) {
    RelearnException::hide_messages = false;

    const std::string message = "This is the first value: {} and this the second: {}\n";
    const std::string expected_message = "This is the first value: 123456 and this the second: false\n";

    try {
        RelearnException::fail(message, 123456, false);
    } catch (const RelearnException& ex) {
        const auto& reason = ex.what();
        ASSERT_EQ(expected_message, reason);
    }

    RelearnException::hide_messages = true;
}

TEST_F(RelearnExceptionTest, testFormattingWrongNumberArguments) {
    ASSERT_ANY_THROW(RelearnException::fail("{}"));
    ASSERT_ANY_THROW(RelearnException::fail("{} {}", 2));
    ASSERT_ANY_THROW(RelearnException::fail("{}", 4.2, false));
    ASSERT_ANY_THROW(RelearnException::fail("{}", "Hallo", 32));
    ASSERT_ANY_THROW(RelearnException::fail("{} {} {}"));
}
